import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

# Step 1: Read the Moby Dick file from the Gutenberg dataset
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Step 2: Tokenization
tokens = word_tokenize(moby_dick)

# Step 3: Stop-words filtering
filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]

# Step 4: Parts-of-Speech (POS) tagging and frequency counting
pos_tags = nltk.pos_tag(filtered_tokens)
pos_freq = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_freq.most_common(5)

print("Most common parts of speech:")
for pos, freq in top_pos:
    print(f"{pos}: {freq}")

# Step 5: Lemmatization
lemmatizer = nltk.WordNetLemmatizer()
from nltk.corpus import wordnet

lemmatized_tokens = []

for word, pos in pos_tags[:20]:
    if pos.startswith('N'):
        pos = wordnet.NOUN
    elif pos.startswith('V'):
        pos = wordnet.VERB
    elif pos.startswith('R'):
        pos = wordnet.ADV
    elif pos.startswith('J'):
        pos = wordnet.ADJ
    else:
        pos = wordnet.NOUN  # 默认词性标记为名词

    lemmatized_word = lemmatizer.lemmatize(word, pos)
    lemmatized_tokens.append(lemmatized_word)

print("Lemmatized tokens (top 20):")
print(lemmatized_tokens)

print("Lemmatized tokens (top 20):")
print(lemmatized_tokens)

# Step 6: Plotting frequency distribution
pos_freq.plot()

# Save the exercise solution to your GitHub repository and enter the URL in the submission box.