from collections import defaultdict, Counter

class HMM:
    def __init__(self, dataset, tags):
        self.dataset = dataset        
        self.tags = tags

        self.tag_data = self.get_tag_data()
        self.states = self.tags

        self.start_counts = defaultdict(int)
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        
        self.update_start_counts()
        self.update_emission_tag_counts()
        
        self.start_probabilities = {}
        self.transition_probabilities = {}
        self.emission_probabilities = {}

    
    def get_tag_data(self):
        tag_data = []

        for sentence in self.dataset:
            tag_list = ['<S>']
            tag_list.extend([tag for _, tag in sentence])
            tag_list.append('<E>')

            tag_data.append(tag_list)

        return tag_data
    
    
    def update_start_counts(self):
        for tags in self.tag_data:
            self.start_counts[tags[1]] += 1

    
    def update_emission_tag_counts(self):
        state_counts = Counter()
        for sentence in self.dataset:
            for word, tag in sentence:
                self.emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1

                state_counts[tag] += 1

        self.most_common_state = state_counts.most_common(1)[0][0]
    
    def calculate_start_probabilities(self):
        total = sum(self.start_counts.values())
        for tag, count in self.start_counts.items():
            self.start_probabilities[tag] = count / total

    def calculate_transition_probabilities(self):
        for tag, next_tags in self.transition_counts.items():
            total = sum(next_tags.values())
            self.transition_probabilities[tag] = {next_tag: count / total for next_tag, count in next_tags.items()}

    def calculate_emission_probabilities(self):
        for tag, word_counts in self.emission_counts.items():
            total = self.tag_counts[tag]
            self.emission_probabilities[tag] = {word: count / total for word, count in word_counts.items()}

    def train(self):
        self.calculate_start_probabilities()
        self.calculate_transition_probabilities()
        self.calculate_emission_probabilities()


    def print_parameters(self):
        print("Start Probabilities:")
        print(self.start_probabilities)
        print("\nTransition Probabilities:")
        print(self.transition_probabilities)
        print("\nEmission Probabilities:")
        print(self.emission_probabilities)


class BigramHMM(HMM):
    def __init__(self, dataset, tags):
        super().__init__(dataset, tags)
        self.update_transition_counts()

    def update_transition_counts(self):
        for tags in self.tag_data:
            for i in range(len(tags) - 1):
                self.transition_counts[tags[i]][tags[i+1]] += 1

    def viterbi(self, words):
        state = []

        for i, word in enumerate(words):
            p = []
            for tag in self.tags:
                if i == 0:
                    curr = '<S>'
                else:
                    curr = state[-1]

                try:
                    transition_p = self.transition_probabilities.get(curr,{}).get(tag, 0)
                    emission_p = self.emission_probabilities.get(tag, {}).get(word, 0)
                    state_p = transition_p * emission_p
                except KeyError:
                    state_p = 0
                
                p.append(state_p)

            try:
                state_max = self.states[max(enumerate(p), key=lambda x: x[1])[0]]
            except ValueError:
                state_max = self.most_common_state
            state.append(state_max)

        return list(zip(words, state))
    

class TrigramHMM(HMM):
    def __init__(self, dataset, tags):
        super().__init__(dataset, tags)
        self.update_transition_counts()
        
    def update_transition_counts(self):
        states = set()
        for tags in self.tag_data:
            if(len(tags) >= 2):
                states.add((tags[0], tags[1]))
            for i in range(len(tags) - 2):
                self.transition_counts[(tags[i], tags[i+1])][tags[i+2]] += 1
                states.add((tags[i], tags[i+1]))
        self.states = list(states)

    def viterbi(self, words):
        state = []

        for i, word in enumerate(words):
            p = []
            for tag in self.tags:
                if i == 0:
                    curr = ('<S>', '<S>')
                else:
                    curr = state[-1]

                try:
                    transition_p = self.transition_probabilities.get(curr,{}).get(tag, 0)
                    emission_p = self.emission_probabilities.get(tag, {}).get(word, 0)
                    state_p = transition_p * emission_p
                except KeyError:
                    state_p = 0
                
                p.append(state_p)

            try:
                state_max = self.tags[max(enumerate(p), key=lambda x: x[1])[0]]
            except ValueError:
                state_max = self.most_common_state
            state.append((curr[1], state_max))

        return list(zip(words, [tag for _, tag in state]))
        
        
