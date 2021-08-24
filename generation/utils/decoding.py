import torch
import torch.nn.functional as F


class SentenceDecoder:
    def __init__(self, model, tokenizer, sentence, device):
        """
        Initialize the SentenceDecoder class.

        :param model:
        :param tokenizer:
        :param sentence:
        :param device:
        """
        model.eval()
        self.tokens = tokenizer.encode(sentence)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def decode_greedy(self, max_len: int = 30) -> str:
        """

        :param max_len: the maximum length of the output tokens
        :return: decoded sentence
        """
        #
        tokens = [2] + self.tokens + [2]
        #
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
        #
        src_mask = self.model.mask_src(src_tensor)
        #
        with torch.no_grad():
            enc_src = self.model.encoder(src_tensor, src_mask)
        # Start the decoded sequence with the BOS token.
        trg_indexes = [2]
        for i in range(max_len):
            #
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0)
            # Move the target tensor to the specified device.
            trg_tensor = trg_tensor.to(self.device)
            # Mask the tensor.
            trg_mask = self.model.mask_trg(trg_tensor)
            #
            with torch.no_grad():
                #
                output, attention = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            #
            pred_token = output.argmax(2)[:, -1].item()
            #
            trg_indexes.append(pred_token)
            if pred_token == 3:
                #
                break
        #
        trg_tokens = ''.join(self.tokenizer.decode(trg_indexes[1:-1]))
        return trg_tokens

    def decode_sampling(self, temperature: float, top_k: int, top_p: float, max_len=30):
        """

        :param temperature:
        :param top_k:
        :param top_p:
        :param max_len:
        :return:
        """
        scr = torch.tensor(self.tokens)
        scr = scr.unsqueeze(0)
        src_mask = self.model.mask_src(scr)
        encoder_output = self.model.encoder(scr, src_mask)
        trg_indexes = [2]
        decoder_input = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
        trg_mask = self.model.mask_trg(decoder_input)
        decoded_tokens = [torch.LongTensor(trg_indexes)]
        for step in range(max_len):
            decoder_output, attention = self.model.decoder(decoder_input, encoder_output, trg_mask, src_mask)
            next_token = self._sample_next_token(dec_out=decoder_output, temperature=temperature,
                                                 top_k=top_k, top_p=top_p)
            decoded_tokens.append(next_token)
            if next_token == 3:
                break
            decoder_input = decoded_tokens[-1].unsqueeze(0).to(self.device)
            trg_mask = self.model.make_trg_mask(decoder_input)
        trg_sentence = ' '.join(self.tokenizer.decode(decoded_tokens[1:-1]))
        return trg_sentence

    @staticmethod
    def filter_top_k_top_p(logits, top_k: int = 0, top_p: float = 0.0, filter_value=-float('Inf')):
        """

        :param logits:
        :param top_k:
        :param top_p:
        :param filter_value:
        :return:
        """
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

    def _sample_next_token(self, dec_out, temperature: float, top_k: int, top_p: float):
        """

        :param dec_out:
        :param temperature:
        :param top_k:
        :param top_p:
        :return:
        """
        logits = dec_out[0, -1, :] / temperature
        filtered_logits = self.filter_top_k_top_p(logits, top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        return next_token

