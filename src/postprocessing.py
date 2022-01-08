from scripts.anntools import Sentence, Keyphrase, Collection, Relation


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
            predicted_label = p.argmax()
            if predicted_label > 0 and predicted_label % 2 == 1:
                word_sequence = [j]
                for k in range(j + 1, len(prediction)):
                    if tokens[k] == '[SEP]':
                        break
                    if prediction[k].argmax() == predicted_label + 1:
                        word_sequence.append(k)
                    else:
                        break
                word_sequence_char_span = (
                    encoded_sentence.token_to_chars(j)[0],
                    encoded_sentence.token_to_chars(word_sequence[-1])[1]
                )
                s.keyphrases.append(Keyphrase(s,
                                              ne_to_label_mapping[predicted_label],
                                              keyphrase_counter,
                                              [word_sequence_char_span]))
                keyphrase_counter += 1
        sentences.append(s)
    return sentences


def extract_relations(predictions, sentences, keyphrases, c: Collection):
    relations = {(i+1): x
                 for i, x in
                 enumerate(["is-a", "same-as", "part-of", "has-property", "causes",
                            "entails", "in-context", "in-place", "in-time",
                            "subject", "target", "domain", "arg",])
                 }
    for p, s, (o, d) in zip(predictions, sentences, keyphrases):
        if p != 0:
            r = Relation(c.sentences[s], o, d, relations[p.item()])
            r_already_in_sentence = False
            for other_r in c.sentences[s].relations:
                if other_r.origin == r.origin and other_r.destination == r.destination and other_r.label == r.label:
                    r_already_in_sentence = True
                    break
            if not r_already_in_sentence:
                c.sentences[s].relations.append(r)
    return c
