import torch
import torch.nn.functional as F




def greedy(model, tokenizer, sentence, device, max_len = 30):

    model.eval()

    tokens = tokenizer.encode(sentence)

    tokens = [2] + tokens + [2]

    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [2]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():



            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[: ,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == 3:
            break

    trg_tokens = ''.join(tokenizer.decode(trg_indexes[1:-1]))

    return trg_tokens


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sampling_next_token(dec_out, temperature, top_k, top_p):
    logits = dec_out[0, -1, :] / temperature
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probabilities = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1)

    return next_token


def sampling(model, tokenizer, sentence, device, temperature, top_k, top_p, max_len=30):
    scr = torch.tensor(tokenizer.encode(sentence))

    scr = scr.unsqueeze(0)

    src_mask = model.make_src_mask(scr)

    encoder_output = model.encoder(scr, src_mask)

    trg_indexes = [2]

    decoder_input = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

    trg_mask = model.make_trg_mask(decoder_input)

    decoded_tokens = [torch.LongTensor(trg_indexes)]
    for step in range(max_len):
        decoder_output, attention = model.decoder(decoder_input, encoder_output, trg_mask, src_mask)
        next_token = sampling_next_token(dec_out=decoder_output, temperature=temperature, top_k=top_k, top_p=top_p)

        decoded_tokens.append(next_token)
        if next_token == 3:
            break

        decoder_input = decoded_tokens[-1].unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(decoder_input)

    trg_sentence = ' '.join(tokenizer.decode(decoded_tokens[1:-1]))

    return trg_sentence