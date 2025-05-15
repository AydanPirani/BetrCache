import pandas as pd
import matplotlib.pyplot as plt

# Read TSV file
l = ["image_id", "caption1", "caption2", "is_cache_hit", "hits_first_record", "llm_similarity", "embedding_similarity"]
df = pd.read_csv('embedding.tsv', sep='\t', names=l)

# Histogram of LLM Similarity
plt.figure()
plt.hist(df['llm_similarity'].dropna(), bins=30)
plt.title('Distribution of LLM Similarity')
plt.xlabel('LLM Similarity')
plt.ylabel('Frequency')
plt.show()

# Histogram of Embedding Similarity
plt.figure()
plt.hist(df['embedding_similarity'].dropna(), bins=30)
plt.title('Distribution of Embedding Similarity')
plt.xlabel('Embedding Similarity')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: LLM vs Embedding Similarity
plt.figure()
plt.scatter(df['llm_similarity'], df['embedding_similarity'])
plt.title('LLM Similarity vs Embedding Similarity')
plt.xlabel('LLM Similarity')
plt.ylabel('Embedding Similarity')
plt.show()

# Bar chart: Cache Hit vs Miss Counts
hit_counts = df['is_cache_hit'].astype(str).value_counts()
total_queries = hit_counts.sum()
hits = hit_counts.get('True', 0)
hit_rate = hits / total_queries
print(f"Cache Hit Rate: {hit_rate:.2%}")

# Boxplot of LLM Similarity by Cache Hit Status
groups = [df[df['is_cache_hit']==val]['llm_similarity'].dropna() for val in sorted(df['is_cache_hit'].unique())]
labels = [str(val) for val in sorted(df['is_cache_hit'].unique())]
plt.figure()
plt.boxplot(groups)
plt.xticks(range(1, len(labels)+1), labels)
plt.title('LLM Similarity by Cache Hit Status')
plt.xlabel('Is Cache Hit')
plt.ylabel('LLM Similarity')
plt.show()