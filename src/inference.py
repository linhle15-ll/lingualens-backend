#inference.py
import collections
import torch


def beam_search_decode(encoder_out, h, c, decoder, vocabulary, max_len, beam_width, block_ngram_size=0):
    # Initialize beam
    initial_token = vocabulary.wordToIdx[vocabulary.start]
    # Beam item: (score, current_sequence, hidden_state, cell_state, attention_weights, set_of_seen_ngrams)
    beam = [(
        0.0, # initial log-probability score
        [initial_token], # current sequence of word indices
        h, c, # initial hidden and cell states
        [], # attention weights
        set() # For storing seen n-grams (of size block_ngram_size) for this path
    )]

    completed_sequences = []

    for step in range(max_len):
        new_beam = []
        for score, seq, h_state, c_state, att_weights, seen_ngrams_current_path in beam:
            last_token = seq[-1]

            # if the sentence is completed (last toke is the <END/>), no more expansion
            if last_token == vocabulary.wordToIdx[vocabulary.end]:
                completed_sequences.append((score, seq, att_weights))
                continue

            emb = decoder.embedding(torch.tensor([last_token]).to(encoder_out.device))
            context, weights = decoder.attention(encoder_out, h_state)

            h_state, c_state = decoder.lstm_cell(torch.cat([emb, context], dim=1), (h_state, c_state))
            logits = decoder.fc(h_state)
            log_probs = F.log_softmax(logits, dim=1)

            # Prepare candidates for expansion
            candidate_log_probs, candidate_tokens = log_probs.topk(decoder.vocab_size, dim=1) # Get all words

            # Apply n-gram blocking
            final_candidates = [] # Will store (log_prob, token_id) tuples
            for i in range(decoder.vocab_size):
                token = candidate_tokens[0, i].item()
                log_prob = candidate_log_probs[0, i].item()

                if block_ngram_size > 1 and len(seq) >= block_ngram_size - 1:
                    potential_ngram = tuple(seq[-(block_ngram_size - 1):]) + (token,)
                    if potential_ngram in seen_ngrams_current_path: # prevent repetition in the result sentence
                        log_prob = -1e9 # Block by assigning a very low probability

                final_candidates.append((log_prob, token))

            # Sort candidates by (potentially penalized) log_prob and take top 'beam_width'
            final_candidates.sort(key=lambda x: x[0], reverse=True)
            selected_top_candidates = final_candidates[:beam_width]


            for candidate_log_prob, candidate_token in selected_top_candidates:
                new_score = score + candidate_log_prob
                new_seq = seq + [candidate_token]
                new_att_weights = att_weights + [weights.squeeze().cpu().reshape(7, 7)]

                # Update seen ngrams for the new sequence
                new_seen_ngrams = set(seen_ngrams_current_path) # Copy the set
                if block_ngram_size > 1 and len(new_seq) >= block_ngram_size:
                    current_ngram = tuple(new_seq[-block_ngram_size:])
                    new_seen_ngrams.add(current_ngram)

                new_beam.append((new_score, new_seq, h_state, c_state, new_att_weights, new_seen_ngrams))


        # Sort all candidates from new_beam and select top 'beam_width'
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]

        # Early stopping if enough completed sequences (optional, for speed)
        # If we have enough completed sequences, we can prune the beam to only contain better ones,
        # or stop early if the best completed sequence is much better than anything in beam.
        # For now, let's keep iterating to max_len to ensure full exploration up to max_len.


    # After loop, combine completed_sequences with remaining sequences in beam
    # (that might not have ended with <END> token yet)
    for score, seq, _, _, att_weights, _ in beam:
        # Ensure the sequence ends with <END> if not already
        if seq[-1] != vocabulary.wordToIdx[vocabulary.end]:
            seq.append(vocabulary.wordToIdx[vocabulary.end])
        completed_sequences.append((score, seq, att_weights))

    # Pick the best sequence from completed_sequences
    if not completed_sequences:
        # If no sequence completed (very rare), return the best from the beam as is
        # This branch needs to handle the case where beam might be empty or only contains partial sequences
        # which is why we add remaining beam items to completed_sequences before this.
        # If still empty, something is wrong.
        return vocabulary.decode([vocabulary.wordToIdx[vocabulary.unk]]), [] # Return UNK if nothing
    else:
        completed_sequences.sort(key=lambda x: x[0], reverse=True)
        best_score, best_seq, best_att_weights = completed_sequences[0]

    return vocabulary.decode(best_seq), best_att_weights

# New ver of generate caption using Beam search
def generate_caption(encoder, decoder, image_tensor, vocabulary, max_len, beam_width=5, block_ngram_size=3):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        image = image_tensor.unsqueeze(0).to(encoder.device)  # (1, 3, 224, 224)
        features = encoder(image)                         # (1, 49, 256)
        h, c = decoder.init_hidden(features)

        # Use beam search decoding
        caption_str, all_weights = beam_search_decode(features, h, c, decoder, vocabulary, max_len, beam_width, block_ngram_size)

        # Correctly remove the UNK token from the caption string
        if vocabulary.unk in caption_str:
            caption_str = caption_str.replace(vocabulary.unk, "").strip()

    return caption_str, all_weights


