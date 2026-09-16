import torch

from config import DEVICE, TOKENIZER_FILE, CHECKPOINT_FILE, MAX_LEN
from tokenizer.tokenizer import Vocabulary
from model.transformer import AURATransformer


def load_model():

    vocab = Vocabulary.load(TOKENIZER_FILE)

    model = AURATransformer(
        vocab_size=len(vocab),
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=MAX_LEN,
        pad_id=vocab.pad_id
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, vocab


@torch.no_grad()
def translate(model, vocab, text):

    src_ids = vocab.encode(text)

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE
    )

    src_padding_mask = src.eq(vocab.pad_id)

    src_emb = model.positional(
        model.embedding(src) * model.d_model ** 0.5
    )

    memory = model.transformer.encoder(
        src_emb,
        src_key_padding_mask=src_padding_mask
    )

    generated = [
        vocab.bos_id
    ]

    for _ in range(MAX_LEN):

        tgt = torch.tensor(
            [generated],
            dtype=torch.long,
            device=DEVICE
        )

        tgt_emb = model.positional(
            model.embedding(tgt) * model.d_model ** 0.5
        )

        tgt_mask = model.generate_causal_mask(
            tgt.size(1),
            tgt.device
        )

        decoder_output = model.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask
        )

        logits = model.output(decoder_output)

        next_token = logits[:, -1, :].argmax(
            dim=-1
        ).item()

        generated.append(next_token)

        if next_token == vocab.eos_id:
            break

    return vocab.decode(generated)


def main():

    print("=" * 50)
    print("AURA v0.1 INFERENCE")
    print("=" * 50)

    model, vocab = load_model()

    print("Device:", DEVICE)
    print("Vocabulary:", len(vocab))
    print()
    print("Type an English phrase.")
    print("Type 'quit' to exit.")
    print()

    while True:

        text = input("AURA> ").strip()

        if text.lower() == "quit":
            break

        if not text:
            continue

        translation = translate(
            model,
            vocab,
            text
        )

        print("Oshindonga:", translation)
        print()


if __name__ == "__main__":
    main()