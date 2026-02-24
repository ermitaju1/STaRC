import os
import numpy as np
import json
import pickle

def load_clip_memory_bank(args):
    first_iter = True
    for bank_type in args.bank_type:
        print("##########################################", bank_type)

        # 경로 설정
        if bank_type in ["yc2", "ViTT", 'anet', "egome"]:
            # import pdb; pdb.set_trace()
            rag = os.path.join(args.bank_path, bank_type)
            filename_prefix = bank_type
            sentence_path = os.path.join(rag, filename_prefix + "_scene_sentences.npy")
        elif bank_type in ["cc3m", "coco"]:
            rag = os.path.join(args.bank_path, bank_type)
            sentence_path = os.path.join(rag, "clip_memory_bank/scene_sentences.json")
        else:
            rag = os.path.join(args.bank_path, 'knowledge')
            filename_prefix = bank_type
            sentence_path = os.path.join(rag, filename_prefix + "_scene_sentences.npy")

        # 텍스트 불러오기 (json or npy)
        if sentence_path.endswith(".json"):
            with open(sentence_path, "r") as f:
                text_sentence = np.array(json.load(f))
        else:
            text_sentence = np.load(sentence_path)

        # 임베딩 불러오기 (모두 npy)
        if bank_type in ["cc3m", "coco"]:
            text_embed = np.load(os.path.join(rag, "clip_memory_bank/clip_token_embeds.npy"))
        else:
            text_embed = np.load(os.path.join(rag, filename_prefix + "_clip_token_embeds.npy"))

        # 첫 번째 반복이면 초기화, 아니면 누적
        if first_iter:
            text_sentences = text_sentence
            text_embeds = text_embed
            first_iter = False
        else:
            text_sentences = np.concatenate((text_sentences, text_sentence))
            text_embeds = np.concatenate((text_embeds, text_embed))

    pair_bank = {
        "vid_sentences": text_sentences,
        "vid_sent_embeds": text_embeds,
        "vide_sent_captions": text_sentences,
    }
    print("memory loaded, ", len(pair_bank["vid_sent_embeds"]))
    return pair_bank

def load_t5_memory_bank(args):
    first_iter = True
    for bank_type in args.bank_type:
        print("########################################## Loading T5 Bank:", bank_type)

        rag = os.path.join(args.bank_path, bank_type)
        filename_prefix = bank_type

        # 1. 텍스트 문장(Caption) 불러오기
        sentence_path = os.path.join(rag, "t5", filename_prefix + "_scene_sentences.json")
        with open(sentence_path, "r") as f:
            # feature_save_t5에서 json.dump(seg_sents, ...)로 저장한 구조 대응
            data = json.load(f)
            text_sentence = np.array(data['text'])

        # 2. T5 임베딩 불러오기 (_t5_embeds.npy)
        embed_path = os.path.join(rag, "t5", filename_prefix + "_t5_embeds.npy")
        text_embed = np.load(embed_path)

        # 3. T5 토큰 ID 불러오기 (_t5_token_ids.pkl)
        # T5는 가변 길이 시퀀스를 cat하여 사용하므로 pkl 로드가 안전합니다.
        token_id_path = os.path.join(rag, "t5", filename_prefix + "_t5_token_ids.pkl")
        with open(token_id_path, 'rb') as f:
            text_token_id = pickle.load(f)

        # 첫 번째 반복이면 초기화, 아니면 누적
        if first_iter:
            text_sentences = text_sentence
            text_embeds = text_embed
            text_token_ids = text_token_id
            first_iter = False
        else:
            text_sentences = np.concatenate((text_sentences, text_sentence))
            text_embeds = np.concatenate((text_embeds, text_embed))
            text_token_ids.extend(text_token_id) # 리스트 형태이므로 extend 사용

    pair_bank = {
        "vid_sentences": text_sentences,
        "vid_sent_embeds": text_embeds,
        "vid_sent_token_ids": text_token_ids, # T5 토큰 ID 추가
        "vide_sent_captions": text_sentences,
    }
    
    print(f"T5 memory loaded. Sentences: {len(pair_bank['vid_sentences'])}, Embeds: {len(pair_bank['vid_sent_embeds'])}")
    return pair_bank