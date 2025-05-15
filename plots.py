import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

threshold = 0.9

# Read TSV file
l = ["image_id", "caption1", "caption2", "is_cache_hit", "hits_first_record", "llm_similarity", "embedding_similarity", "act_llm_similarity", "act_embedding_similarity", "latency"]
df = pd.read_csv(f"outputs/embedding_{threshold}.tsv", sep='\t', names=l)

# Histogram of LLM Similarity
plt.figure()
plt.hist(df['llm_similarity'].dropna(), bins=30)
plt.title('Distribution of LLM Similarity')
plt.xlabel('LLM Similarity')
plt.ylabel('Frequency')
plt.savefig(f"plots/{threshold}_llm_similarity.png")

# Histogram of Embedding Similarity
plt.figure()
plt.hist(df['embedding_similarity'].dropna(), bins=30)
plt.title('Distribution of Embedding Similarity')
plt.xlabel('Embedding Similarity')
plt.ylabel('Frequency')
plt.savefig(f"plots/{threshold}_embedding_similarity.png")

# Scatter plot: LLM vs Embedding Similarity
plt.figure()
plt.scatter(df['llm_similarity'], df['embedding_similarity'])
plt.title('LLM Similarity vs Embedding Similarity')
plt.xlabel('LLM Similarity')
plt.ylabel('Embedding Similarity')
plt.savefig(f"plots/{threshold}_scatter.png")

# Bar chart: Cache Hit vs Miss Counts
hit_counts = df['is_cache_hit'].astype(str).value_counts()
total_queries = hit_counts.sum()
hits = hit_counts.get('True', 0)
hit_rate = hits / total_queries
print(f"Cache Hit Rate: {hit_rate:.2%}")

# Boxplot of LLM Similarity by Cache Hit Status
groups = [df[df['is_cache_hit']==val]['llm_similarity'].dropna() for val in sorted(df['is_cache_hit'].unique())]
labels = [str(val) for val in sorted(df['is_cache_hit'].unique())]
# plt.figure()
# plt.boxplot(groups)
# plt.xticks(range(1, len(labels)+1), labels)
# plt.title('LLM Similarity by Cache Hit Status')
# plt.xlabel('Is Cache Hit')
# plt.ylabel('LLM Similarity')
# plt.show()

for label, group in zip(labels, groups):
    q1 = np.percentile(group, 25)
    median = np.median(group)
    q3 = np.percentile(group, 75)
    iqr = q3 - q1
    lower_whisker = max(min(group), q1 - 1.5 * iqr)
    upper_whisker = min(max(group), q3 + 1.5 * iqr)
    
    print(f"Group: {label}")
    print(f"Min: {lower_whisker:.4f}, Q1: {q1:.4f}, Median: {median:.4f}, Q3: {q3:.4f}, Max: {upper_whisker:.4f}\n")