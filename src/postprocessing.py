from scripts.anntools import Sentence, Keyphrase, Collection


def extract_named_entities(tokenizer, predictions, input_tokens, c: Collection):
    ne_to_label_mapping = {
        1: 'Action',
        3: 'Concept',
        5: 'Predicate',
        7: 'Reference'
    }
    sentences = []
    keyphrase_counter = 0
    for i, (tokens, prediction, og_sentence) in enumerate(
            zip([tokenizer.convert_ids_to_tokens(x) for x in input_tokens['input_ids']],
                predictions,
                c.sentences)):

        s = Sentence(og_sentence.text)
        encoded_sentence = input_tokens[i]

        for j, p in enumerate(prediction):
            if tokens[j] == '[CLS]':
                continue
            if tokens[j] == '.' or tokens[j] == '[SEP]':
                break
            # If prediction[j] is B token, get the entire word
            if p > 0 and p % 2 == 1:
                word_sequence = [j]
                for k in range(j + 1, len(prediction)):
                    if tokens[k] == '[SEP]':
                        break
                    if prediction[k] == p + 1:
                        word_sequence.append(k)
                    else:
                        break
                word_sequence_char_span = (
                    encoded_sentence.token_to_chars(j)[0],
                    encoded_sentence.token_to_chars(word_sequence[-1])[1]
                )
                s.keyphrases.append(Keyphrase(s,
                                              ne_to_label_mapping[p],
                                              keyphrase_counter,
                                              [word_sequence_char_span]))
                keyphrase_counter += 1
        sentences.append(s)
    return sentences
