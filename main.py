import json
import math
import re
from unidecode import unidecode
from parsivar import FindStems
from hazm import *
import time

N = 5000
r = 20  # for champions list


class Tokenizer(object):
    def __init__(self):
        self.tokens = []
        self.dictionary = {}
        self.stop_words = []

    def tokenize_doc(self, doc):
        dot_split = doc['content'].split(".")
        dot_split = dot_split[:-1]  # remove the last phrase: انتهای پیام
        removed_new_line = [re.sub(r'\n', ' ', text) for text in dot_split]
        removed_colons = [re.sub(r':', ' ', text) for text in removed_new_line]
        removed_commas = [re.sub(r'،', ' ', text) for text in removed_colons]
        removed_signs = [re.sub(r'[!؟<>()«»|]', ' ', text) for text in removed_commas]
        english_numbers = [self.persian_number_to_english(text) for text in removed_signs]
        corrected_verbs = [self.correct_semi_spaces(text) for text in english_numbers]

        # TODO: do better tokenization
        tokens = {}
        position = 0
        for text in corrected_verbs:
            text = text.split()
            for token in text:

                stemmed_token = self.stemming(token)
                # if token != stemmed_token:
                #     print(f"original: {token}, stemmed: {stemmed_token}")
                if stemmed_token not in tokens.keys():
                    tokens[stemmed_token] = {'count': 1, 'positions': [position]}
                else:
                    tokens[stemmed_token]['count'] += 1
                    tokens[stemmed_token]['positions'].append(position)
                position += 1
        # print(tokens)  # {'word' : {'count':2, 'positions' = [3, 49]}
        return tokens

    def correct_semi_spaces(self, text):
        modified_text = re.sub(r' می ', ' می\u200c', text)
        modified_text = re.sub(r' نمی ', ' نمی\u200c', modified_text)
        modified_text = re.sub(r' ای ', '\u200cای ', modified_text)
        modified_text = re.sub(r' ها ', '\u200cها ', modified_text)
        modified_text = re.sub(r' های ', '\u200cهای ', modified_text)
        modified_text = re.sub(r' هایی ', '\u200cهایی ', modified_text)
        modified_text = re.sub(r' تر ', '\u200cتر ', modified_text)
        modified_text = re.sub(r' تری ', '\u200cتری ', modified_text)
        modified_text = re.sub(r' ترین ', '\u200cترین ', modified_text)
        modified_text = re.sub(r' گر ', '\u200cگر ', modified_text)
        modified_text = re.sub(r' گری ', '\u200cگری ', modified_text)
        modified_text = re.sub(r' ام ', '\u200cام ', modified_text)
        modified_text = re.sub(r' ات ', '\u200cات ', modified_text)
        modified_text = re.sub(r' اش ', '\u200cاش ', modified_text)
        return modified_text

    def persian_number_to_english(self, token: str):
        persian_digits = '۰۱۲۳۴۵۶۷۸۹'
        english_digits = '0123456789'
        translation_table = str.maketrans(persian_digits, english_digits)
        english_number = token.translate(translation_table)
        return english_number

    def stemming(self, token: str):
        # my_stemmer = FindStems()
        # return my_stemmer.convert_to_stem(token)
        stemmer = Stemmer()
        return stemmer.stem(token)

    def make_inverted_index(self, dataset):
        for doc_id, doc in dataset.items():
            print(f"doc_id: {doc_id}")
            # print("content:", doc["content"])
            tokens = self.tokenize_doc(doc)
            for token in tokens:
                if token not in self.dictionary.keys():
                    self.dictionary[token] = {'total_count': tokens[token]['count'],
                                              'docs': [(doc_id, tokens[token])]}
                else:
                    self.dictionary[token]['total_count'] += tokens[token]['count']
                    self.dictionary[token]['docs'].append((doc_id, tokens[token]))
            # print(self.dictionary)
        return self.dictionary

    # stop words: ['و', 'در', 'به', 'از', 'این', 'که', 'با', 'را', 'است', 'برای', 'کرد', 'هم', 'تیم', 'ما', 'شد', 'یک', 'آن', 'بود', 'باید', 'تا', 'کشور', 'وی', 'بر', 'بازی', 'شده', 'خود', 'مجلس', 'اسلامی', 'گفت', 'فارس', 'مردم', 'گزارش', 'ایران', 'خبرگزاری', 'اما', 'دولت', 'شود', 'داشت', 'دارد', 'سال', 'ملی', 'اینکه', 'قرار', 'دو', 'رئیس', 'کند', 'می\u200cشود', 'کار', 'نیز', 'امروز']
    def remove_stop_words(self):
        frequency_sorted_dictionary = sorted(self.dictionary.items(), key=lambda x: x[1]["total_count"], reverse=True)
        stop_words = [item[0] for item in frequency_sorted_dictionary[:50]]
        for word in stop_words:
            del self.dictionary[word]
        self.stop_words = stop_words


class SearchEngine:
    def __init__(self, dataset: dict, inverted_index: dict):
        self.inverted_index = inverted_index
        self.dataset = dataset
        self.documents_vectors = {}
        self.tokenizer = Tokenizer()
        # self.champions_list = champions_list

    def vectorize_doc(self, doc: dict) -> dict:
        # tfidf = (1 + log tf) * log (N/nt)
        vector = {}
        for token, details in self.tokenizer.tokenize_doc(doc).items():
            if token in self.inverted_index.keys():
                tf = 1 + math.log10(details['count'])
                idf = math.log10(N / len(self.inverted_index[token]['docs']))
                vector[token] = tf * idf
        return vector

    def vectorize_docs(self):
        for doc_id, doc in self.dataset.items():
            self.documents_vectors[doc_id] = self.vectorize_doc(doc)
        # print(self.documents_vectors)
        return self.documents_vectors

    def vectorize_query(self, query: str):
        vector = {}
        words = query.split()
        for word in words:
            if word in self.inverted_index:
                tf = 1 + math.log10(query.count(word))
                idf = math.log10(N / len(self.inverted_index[word]['docs']))
                vector[word] = tf * idf
        # print(vector)
        return vector

    def cosine_similarity(self, doc_vector: dict, query_vector: dict):
        numerator = 0
        for word, score in query_vector.items():
            if word in doc_vector.keys():
                numerator += doc_vector[word] * score
        try:
            similarity = numerator / (
                    sum(x ** 2 for x in doc_vector.values()) * sum(x ** 2 for x in query_vector.values()))
        except ZeroDivisionError:
            similarity = 0
        # print(similarity)
        return similarity

    def find_k_docs(self, query_vector: dict, k: int):
        cosine_similarities = {}
        other_documents_vectors = self.documents_vectors.copy()
        # del other_documents_vectors['4092']
        for doc_id, doc_vector in other_documents_vectors.items():
            cosine_similarities[doc_id] = self.cosine_similarity(doc_vector, query_vector)
        sorted_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_similarities[:k])
        return [item[0] for item in sorted_similarities[:k]]

    def index_elimination(self, query_vector: dict, k: int, method: str):
        cosine_similarities = {}

        if method == 'simple':
            for word in query_vector.keys():
                # get documents that have at least one word of the query
                related_docs = self.inverted_index[word]["docs"]
                for doc in related_docs:
                    cosine_similarities[doc[0]] = self.cosine_similarity(self.documents_vectors[doc[0]], query_vector)

        if method == 'advanced':
            related_docs = {}
            # get documents that have all the words in the query
            for word in query_vector.keys():
                related_docs[word] = [doc[0] for doc in self.inverted_index[word]["docs"]]
            docs_containing_all_words = self.intersection(related_docs)[0]
            # print(docs_containing_all_words)
            for doc in docs_containing_all_words:
                cosine_similarities[doc] = self.cosine_similarity(self.documents_vectors[doc], query_vector)

        sorted_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_similarities[:k]]

    def intersection(self, related_docs):
        lists = list(related_docs.values())
        if len(lists) < 2:
            return lists
        intersection_result = set(lists[0]).intersection(*lists[1:])
        result_list = list(intersection_result)
        return result_list

    def create_champions_list(self):
        champions_list = {}
        for token, details in self.inverted_index.items():
            # print("token:", token)
            tf = {}
            all_docs = details["docs"]
            for doc in all_docs:
                tf[doc[0]] = doc[1]['count']
            sorted_tf = sorted(tf.items(), key=lambda x: x[1], reverse=True)
            champions_list[token] = [item[0] for item in sorted_tf[:r]]
        self.champions_list = champions_list
        return champions_list

    def search_in_champions_list(self, query_vector: dict, k: int):
        cosine_similarities = {}
        for word in query_vector.keys():
            related_docs = self.champions_list[word]
            for doc in related_docs:
                cosine_similarities[doc] = self.cosine_similarity(self.documents_vectors[doc], query_vector)
        sorted_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_similarities[:k]]

    def print_top_k_docs(self, doc_ids: list) -> None:
        counter = 1
        for doc_id in doc_ids:
            print(f"{counter}) doc id:{doc_id}")
            print(f"title:{self.dataset[doc_id]['title']}")
            print(f"url:{self.dataset[doc_id]['url']}")
            # print(f"content:{self.dataset[doc_id]['content']}")
            print("**********************************")
            counter += 1

    def longest_postings_lists(self):
        docs_count = {}
        for token, details in self.inverted_index.items():
            docs_count[token] = len(details["docs"])
        sorted_long_postings_lists = sorted(docs_count.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_long_postings_lists)

    def sort_based_on_idf(self):
        docs_idf = {}
        for token, details in self.inverted_index.items():
            docs_idf[token] = math.log10(N / len(self.inverted_index[token]['docs']))
        sorted_idf = sorted(docs_idf.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_idf)

    def first_doc_analysis(self):
        vector = self.vectorize_doc(self.dataset.get('4092'))
        print(vector)
        sorted_vector = sorted(vector.items(), key=lambda x: x[1], reverse=True)

    def find_most_similar_doc(self):
        self.vectorize_docs()
        similar_doc_id = self.find_k_docs(self.documents_vectors.get("4092"), 1)
        # print(similar_doc_id)

    def find_doc_stop_words(self, doc_id):
        doc = self.dataset.get(doc_id)
        tokens = self.tokenizer.tokenize_doc(doc)
        for token in tokens:
            if token not in self.inverted_index.keys():
                print(token)


if __name__ == '__main__':
    f = open('IR_data_news_12k.json')
    dataset = json.load(f)
    tokenizer = Tokenizer()
    inverted_index = tokenizer.make_inverted_index(dataset)
    tokenizer.remove_stop_words()
    # print(inverted_index)

    # doc = dataset.get("4092")
    # tokens = tokenizer.tokenize_doc(doc)
    # not_deleted = []
    # for token in tokens:
    #     if token not in tokenizer.stop_words:
    #         not_deleted.append(token)
    # print(not_deleted)

    # with open("inverted_index_12k.json", "w") as file:
    #     json.dump(inverted_index, file)
    # print("done")

    # f2 = open('inverted_index_12k.json')
    # inverted_index = json.load(f2)

    # f3 = open('champions_list.json')
    # champions_list = json.load(f3)

    search_engine = SearchEngine(dataset, inverted_index)

    # METHODS QUESTIONED IN THE PROJECT EXAM
    # search_engine.find_doc_stop_words("4092")
    # search_engine.longest_postings_lists()
    # search_engine.sort_based_on_idf()
    # search_engine.first_doc_analysis()
    # search_engine.find_most_similar_doc()

    search_engine.vectorize_docs()
    query_vector = search_engine.vectorize_query('داوران فوتبال')
    best_k_docs = search_engine.find_k_docs(query_vector, 5)
    search_engine.print_top_k_docs(best_k_docs)

    # print(f"simple search start: {time.time()}")
    # best_k_docs_index_elimination = search_engine.index_elimination(query_vector, 5, 'advanced')
    # search_engine.print_top_k_docs(best_k_docs_index_elimination)
    # search_engine.print_top_k_docs(best_k_docs)
    # print(f"simple search end: {time.time()}")
    # champions_list = search_engine.create_champions_list()
    # print(champions_list)
    # with open("champions_list.json", "w") as file:
    #     json.dump(champions_list, file)
    # print("done")
    # print("***************")
    # print(f"champions search start: {time.time()}")
    # result_champions_list = search_engine.search_in_champions_list(query_vector, 10)
    # search_engine.print_top_k_docs(result_champions_list)
    # print(f"champions search end: {time.time()}")
