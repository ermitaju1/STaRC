import torch
import torch.nn as nn
from .modeling_t5 import T5ForConditionalGeneration
from .vit import VisionTransformer
from transformers import T5Tokenizer
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from util.HD_loss import loss_saliency
import numpy as np
import torch.nn.functional as F
import random, math
# from torch.nn.functional import cosine_similarity
from .asot import *
from time import time

def _get_tokenizer(tokenizer_path, num_bins=0):
    if 't5' in tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if num_bins:
            new_tokens = ["<time=" + str(i) + ">" for i in range(num_bins)]
            tokenizer.add_tokens(list(new_tokens))
    else:
        raise NotImplementedError(tokenizer_path)
    return tokenizer

class STaRC(torch.nn.Module):
    def __init__(self,
                 t5_path,
                 num_features=100,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=2048,
                 vis_drop=0.,
                 tokenizer=None,
                 enc_drop=0.,
                 dec_drop=0.1,
                 use_speech=True,
                 use_video=True,
                 num_bins=100,
                 label_smoothing=0.1,
                 args=None):
        super().__init__()
        self.args = args
        self.t5_model = T5ForConditionalGeneration.from_pretrained(encoder_dropout=enc_drop, decoder_dropout=dec_drop, label_smoothing=label_smoothing,
                                                                   pretrained_model_name_or_path=t5_path, local_files_only=True, is_gated_act="v1_1" in t5_path)
        self.t5_model.resize_token_embeddings(len(tokenizer) - num_bins)
        self.t5_model.resize_token_embeddings(len(tokenizer))
        self.visual_encoder = VisionTransformer(num_features=num_features,
                                                embed_dim=embed_dim,
                                                depth=depth,
                                                num_heads=heads,
                                                mlp_dim=mlp_dim,
                                                qkv_bias=True,
                                                qk_scale=None,
                                                drop_rate=vis_drop,
                                                attn_drop_rate=vis_drop,
                                                norm_layer=nn.LayerNorm)
        self.t5_tokenizer = tokenizer
        self.use_speech = use_speech
        self.use_video = use_video
        self.proj_v2t = nn.Linear(embed_dim, embed_dim)
        self.proj_r2t = nn.Linear(embed_dim, embed_dim)

        hidden_dim = 768
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.saliency_tok_proj = nn.Linear(1, self.t5_model.model_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.window = args.window
        if self.window == 8:
            self.window_sizes = [8, 32, 64] #, 32, 64]
        self.K = args.asot_K
        anchors = torch.randn(self.K, embed_dim)
        self.anchors = nn.Parameter(F.normalize(anchors, dim=-1))
        self.loss_lambda = int(self.args.loss_lambda * 0.5)


    def retrieval(self, video_list, memory_bank, args, anchor_ids_list=None):
        soft_k = args.soft_k

        vocab_features = torch.tensor(memory_bank['vid_sent_embeds'], device=video_list[0].device).float()
        sentences = memory_bank['vide_sent_captions']
        vocab_features = vocab_features.squeeze(1)
        vocab_features = F.normalize(vocab_features, dim=-1)

        if hasattr(args, 'subset_ratio') and args.subset_ratio < 1.0:
            total = vocab_features.size(0)
            subset_size = max(1, int(total * args.subset_ratio))
            indices = random.sample(range(total), subset_size)
            vocab_features = vocab_features[indices]
            sentences = [sentences[i] for i in indices]

        retrieved_sentences_all = []
        retrieved_embeds_list = []
        retrieved_anchor_ids_all = [] 

        for b, video in enumerate(video_list):
            video = F.normalize(video, dim=-1)
            scores = torch.matmul(video, vocab_features.T)
            _, indices = torch.topk(scores, k=soft_k, dim=-1)
            retrieved_sentences = [[sentences[idx] for idx in ind.tolist()] for ind in indices]
            retrieved_sentences_all.append(retrieved_sentences)
            topk_embeds = vocab_features[indices]
            retrieved_embeds = topk_embeds.mean(dim=1)
            retrieved_embeds_list.append(retrieved_embeds)
            if anchor_ids_list is not None:
                retrieved_anchor_ids_all.append(anchor_ids_list[b])


        lengths = torch.tensor([e.size(0) for e in retrieved_embeds_list])
        max_len = max(lengths.tolist())
        padded_embeds = torch.stack([
            F.pad(e, (0, 0, 0, max_len - e.size(0))) for e in retrieved_embeds_list
        ], dim=0)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

        if anchor_ids_list is not None:
            return padded_embeds, mask ,retrieved_sentences_all, retrieved_anchor_ids_all
        else:
            return padded_embeds, mask, None, None


    def forward(self, video, input_tokenized, output_tokenized, timestamp, duration, sal_target = None, mode='None',uns_video=None, memory_bank=None, epoch=0):
        if self.use_video:
            if isinstance(video, dict):  # cached
                video, video_origin, atts_vis = video["video"], video["video_origin"].clone(), video["atts_vis"] 
            else:
                video_origin = video.clone()
                video = self.visual_encoder(video) 
                if self.proj_v2t is not None:
                    video = self.proj_v2t(video)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)
            
            B, T, D = video.shape
            video_clone = video.clone()

            if self.args.self_attn:
                vid_zero = torch.zeros_like(video)
                count = torch.zeros(B, T, 1, device=video.device)

                for window in self.window_sizes:
                    for batch_idx in range(B):
                        sample_not_ln = video[batch_idx]  # [T, D]
                        sample = video[batch_idx]

                        for i in range(0, T - window + 1, window):
                            mini_batch = sample[i : i + window, :]  # [window, D]
                            mini_batch_not_ln = sample_not_ln[i : i + window, :]
                            q = k = mini_batch_not_ln
                            v = mini_batch.clone()
                            scale = D**0.5

                            attn_scores = torch.matmul(q, k.T) / scale  # [window, window]
                            attn_weights = F.softmax(
                                attn_scores, dim=-1
                            )  # [window, window]
                            attn_output = torch.matmul(attn_weights, v)  # [window, D]

                            vid_zero[batch_idx, i : i + window, :] += attn_output
                            count[batch_idx, i : i + window, :] += 1


                vid_zero = vid_zero / (count + 1e-6)
                vid_zero = self.norm(vid_zero)
                video_clone = video + vid_zero


            video_dict = {"video": video_clone, "video_origin": video_origin, "atts_vis": atts_vis} 

            if self.args.use_saliency or self.args.use_ret: 
                video_, video_global = self.visual_encoder.forward_with_global(video_clone, mode = "training")  # B T D

                saliency_score = (
                    torch.sum(self.saliency_proj1(video_clone) * self.saliency_proj2(video_global).unsqueeze(1), dim=-1)
                    / np.sqrt(self.hidden_dim)
                )  # [B, T]

                
                saliency_scores = {"saliency_scores": saliency_score, "video_mask": atts_vis}

                s_n = (saliency_score - saliency_score.mean(dim=1, keepdim=True)) \
                    / (saliency_score.std(dim=1, keepdim=True) + 1e-6)
                w = torch.sigmoid(s_n)  

                video_segs, seg_scores, assign_list, frames_by_anchor_list, segments_meta_list = asot_segments_aux(
                    video_origin,   # [B,T,D]
                    w,             # [B,T]
                    atts_vis,      # [B,T]
                    self.args,
                    self.anchors,  
                    asot_mode="train"

                )

                topk = self.args.asot_topk_for_retrieval

                video_list = []
                anchor_ids_list = []  

                for b in range(len(video_segs)):
                    emb = video_segs[b]        # [N_seg, D]
                    scr = seg_scores[b]        # [N_seg]
                    metas = segments_meta_list[b] 
                    if emb.dim() == 1: 
                        emb = emb.unsqueeze(0)
                    k = min(topk, emb.size(0))

                    top_idx = torch.topk(scr, k=k, dim=0).indices  # [k]
                    video_list.append(emb[top_idx])                # [k, D]

                    anchor_ids = [int(metas[i]["anchor"]) for i in top_idx.tolist()]
                    anchor_ids_list.append(anchor_ids)

            if self.args.use_ret:
                retrieval_emb, ret_mask, _, _ = self.retrieval(video_list, memory_bank, self.args)
                retrieval_emb = self.proj_r2t(retrieval_emb)
                atts_ret = torch.ones(retrieval_emb.size()[:-1], dtype=torch.long).to(video.device)

            if self.args.use_salip and self.args.use_saliency:
                saliency_tok = self.saliency_tok_proj(saliency_score.unsqueeze(-1))  # [B, T, D]
                atts_sal = torch.ones(saliency_tok.size()[:-1], dtype=torch.long, device=video.device)
                video_input = torch.cat([video, saliency_tok], dim=1)                # [B, 2T, D]
                atts_vis_input = torch.cat([atts_vis, atts_sal], dim=1)

            else:
                video_input = video.clone()
                atts_vis_input = atts_vis.clone() 


        else:
            video_dict = None

        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])  # B L D
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        if self.use_video and self.use_speech:
            if self.args.use_ret:
                encoded.last_hidden_state = torch.cat([video_input, retrieval_emb, encoded.last_hidden_state], dim=1)
                encoder_atts = torch.cat([atts_vis_input, atts_ret, input_tokenized['attention_mask']], dim=1)
            else:
                encoded.last_hidden_state = torch.cat([video_input, encoded.last_hidden_state], dim=1)
                encoder_atts = torch.cat([atts_vis_input, input_tokenized['attention_mask']], dim=1)

        elif self.use_video: 
            hidden_state = torch.cat([video_input, retrieval_emb], dim=1)
            encoded = BaseModelOutput(last_hidden_state=hidden_state)
            encoder_atts = torch.cat([atts_vis_input, atts_ret], dim=1)
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        targets = output_tokenized['input_ids'].masked_fill(
            output_tokenized['input_ids'] == self.t5_tokenizer.pad_token_id, -100
        )
        outputs = self.t5_model(
            encoder_outputs=encoded,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokenized['attention_mask'],
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        if sal_target is not None:
            sal_loss = loss_saliency(self.args, saliency_scores, sal_target)
            loss += sal_loss * self.loss_lambda

        return {"loss": loss}, video_dict

    @torch.no_grad()
    def generate(
            self,
            video,
            input_tokenized,
            use_nucleus_sampling=False,
            num_beams=4,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
            memory_bank=None, 
            gt_duration=None,
            sal_target=None,
    ):
        """
        Args:
            video (torch.Tensor): A tensor of shape (batch_size, T, D)
            input_tokenized (torch.Tensor): A tensor of shape (batch_size, L)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        video_origin = video.clone()
        video = self.visual_encoder(video)  # B T D
        if self.proj_v2t is not None:
            video = self.proj_v2t(video)
        atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)
        atts_vis_input = atts_vis.clone()

        B, T, D = video.shape
        video_clone = video.clone()
        if self.args.self_attn:
            vid_zero = torch.zeros_like(video)
            count = torch.zeros(B, T, 1, device=video.device)

            for window in self.window_sizes:
                for batch_idx in range(B):
                    sample_not_ln = video[batch_idx]  # [T, D]
                    sample = video[batch_idx]

                    for i in range(0, T - window + 1, window):
                        mini_batch = sample[i : i + window, :]  # [window, D]
                        mini_batch_not_ln = sample_not_ln[i : i + window, :]
                        q = k = mini_batch_not_ln
                        v = mini_batch.clone()
                        scale = D**0.5

                        attn_scores = torch.matmul(q, k.T) / scale  # [window, window]
                        attn_weights = F.softmax(
                            attn_scores, dim=-1
                        )  # [window, window]
                        attn_output = torch.matmul(attn_weights, v)  # [window, D]

                        vid_zero[batch_idx, i : i + window, :] += attn_output
                        count[batch_idx, i : i + window, :] += 1


            vid_zero = vid_zero / (count + 1e-6)
            vid_zero = self.norm(vid_zero)
            video_clone = video + vid_zero

        if self.args.use_saliency or self.args.use_ret:
            video_, video_global = self.visual_encoder.forward_with_global(video_clone)  # B T D
            saliency_score = (
                torch.sum(self.saliency_proj1(video_clone) * self.saliency_proj2(video_global).unsqueeze(1), dim=-1)
                / np.sqrt(self.hidden_dim)
            )  # [B, T]

            s_n = (saliency_score - saliency_score.mean(dim=1, keepdim=True)) \
                / (saliency_score.std(dim=1, keepdim=True) + 1e-6)
            w = torch.sigmoid(s_n)   # [B, T]

            video_segs, seg_scores, assign_list, anchor_info, segments_meta_list = asot_segments_aux(
                video_origin,   # [B,T,D]
                w,             # [B,T]
                atts_vis,      # [B,T]
                self.args,
                self.anchors,
                asot_mode="infer"
            )

            segment_frames_list = anchor_info[0]
            segment_anchor_ids_list = anchor_info[1]

            topk = getattr(self.args, "asot_topk_for_retrieval", 8)

            video_list = []                              
            selected_frames_per_batch = []               
            selected_anchor_ids_per_batch = []           

            for b in range(len(video_segs)):
                emb = video_segs[b]                      # [N_seg, D]
                scr = seg_scores[b]                      # [N_seg]
                if emb.dim() == 1: 
                    emb = emb.unsqueeze(0)
                k = min(topk, emb.size(0))

                top_idx = torch.topk(scr, k=k, dim=0).indices  # [k]

                video_list.append(emb[top_idx])                # [k, D]

                sel_anchor_ids = segment_anchor_ids_list[b] #[i]  for i in top_idx.tolist()]
                sel_frames     = segment_frames_list[b] #[i]     for i in top_idx.tolist()]

                selected_anchor_ids_per_batch.append(sel_anchor_ids)
                selected_frames_per_batch.append(sel_frames)

        if self.args.use_ret:
            retrieval_emb, ret_mask, sentences, sent_idx  = self.retrieval(video_list, memory_bank, self.args, anchor_ids_list=None) 
            retrieval_emb = self.proj_r2t(retrieval_emb)
            atts_ret = torch.ones(retrieval_emb.size()[:-1], dtype=torch.long).to(video.device)


        if self.args.use_salip and self.args.use_saliency:
            saliency_tok = self.saliency_tok_proj(saliency_score.unsqueeze(-1))  # [B, T, D]
            atts_sal = torch.ones(saliency_tok.size()[:-1], dtype=torch.long, device=video.device)
            video_input = torch.cat([video_, saliency_tok], dim=1)                # [B, 2T, D]
            atts_vis_input = torch.cat([atts_vis, atts_sal], dim=1)

        else:
            video_input = video_.clone()
            atts_vis_input = atts_vis.clone() 

        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])  # B L D
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        if self.use_video and self.use_speech:
            if self.args.use_ret:
                encoded.last_hidden_state = torch.cat([video_input, retrieval_emb, encoded.last_hidden_state], dim=1)
                encoder_atts = torch.cat([atts_vis_input, atts_ret, input_tokenized['attention_mask']], dim=1)
            else:
                encoded.last_hidden_state = torch.cat([video_input, encoded.last_hidden_state], dim=1)
                encoder_atts = torch.cat([atts_vis_input, input_tokenized['attention_mask']], dim=1)
        elif self.use_video:
            hidden_state = torch.cat([video_input, retrieval_emb], dim=1)
            encoded = BaseModelOutput(last_hidden_state=hidden_state)
            encoder_atts = torch.cat([atts_vis_input, atts_ret], dim=1)
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        outputs = self.t5_model.generate(
                encoder_outputs=encoded,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
        )
        
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text
