import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def similarity_vs_hitrate(thresholds):
    plt.figure()

    hitrates = []
    similarities = []

    for t in thresholds:
        l = ["image_id", "caption1", "caption2", "is_cache_hit", "hits_first_record", "cache_llm_similarity", "cache_embedding_similarity", "true_llm_similarity", "true_embedding_similarity", "latency"]
        df = pd.read_csv(f"outputs/embedding_{t}.tsv", sep='\t', names=l)

        hit_counts = df['is_cache_hit'].astype(str).value_counts()
        total_queries = hit_counts.sum()
        hits = hit_counts.get('True', 0)
        hit_rate = hits / total_queries
        hitrates.append(hit_rate)

        average_similarity = df['true_llm_similarity'].mean()
        similarities.append(average_similarity)
    print(similarities)

    plt.plot(thresholds, similarities, color='blue', linestyle='-', linewidth=2, marker='o', label='LLM-as-a-Judge Similarity')
    plt.plot(thresholds, hitrates, color='red', linestyle='-', linewidth=2, marker='o', label='Cache Hit Rate')
    plt.title("Similarity vs Cache Hit Rate")
    plt.xlabel("Threshold Value")
    plt.ylabel("Score/Rate (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("plots/sim_vs_hitrate.png")

def overlapped_pdf(thresholds):
    plt.figure()

    for t in thresholds:
        l = ["image_id", "caption1", "caption2", "is_cache_hit", "hits_first_record", "cache_llm_similarity", "cache_embedding_similarity", "true_llm_similarity", "true_embedding_similarity", "latency"]
        df = pd.read_csv(f"outputs/embedding_{t}.tsv", sep='\t', names=l)

        sns.kdeplot(df['cache_llm_similarity'], common_norm=True, label=f"Threshold={t}")

    plt.title("Cache LLM-as-a-Judge Similarity Distribution")
    plt.xlabel("Cache LLM-as-a-Judge Similarity")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.ylim(0, 13)
    plt.legend()
    plt.savefig(f"plots/overlapped_cache_llm_similarity_pdf.png")

def pdf_plot(df):
    plt.figure()
    sns.kdeplot(df['cache_llm_similarity'], fill=True, common_norm=True)
    plt.title("Cache LLM-as-a-Judge Similarity Distribution")
    plt.xlabel("Cache LLM-as-a-Judge Similarity")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.ylim(0, 13)
    plt.savefig(f"plots/{threshold}_cache_llm_similarity_pdf.png")

    plt.figure()
    sns.kdeplot(df['cache_embedding_similarity'], fill=True, common_norm=True)
    plt.title("Cache Embedding Similarity Distribution")
    plt.xlabel("Cache Embedding Similarity")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.ylim(0, 13)
    plt.savefig(f"plots/{threshold}_cache_embedding_similarity_pdf.png")

    plt.figure()
    sns.kdeplot(df['true_llm_similarity'], fill=True, common_norm=True)
    plt.title("LLM-as-a-Judge Similarity Distribution")
    plt.xlabel("LLM-as-a-Judge Similarity")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.ylim(0, 13)
    plt.savefig(f"plots/{threshold}_llm_similarity_pdf.png")

    plt.figure()
    sns.kdeplot(df['true_embedding_similarity'], fill=True, common_norm=True)
    plt.title("Embedding Similarity Distribution")
    plt.xlabel("Embedding Similarity")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.ylim(0, 13)
    plt.savefig(f"plots/{threshold}_embedding_similarity_pdf.png")


def histogram_plot(df):
    plt.figure()
    plt.hist(df['cache_llm_similarity'].dropna(), bins=30)
    plt.title('Distribution of LLM Similarity')
    plt.xlabel('LLM Similarity')
    plt.ylabel('Frequency')
    plt.savefig(f"plots/{threshold}_llm_histogram.png")

    plt.figure()
    plt.hist(df['cache_embedding_similarity'].dropna(), bins=30)
    plt.title('Distribution of Embedding Similarity')
    plt.xlabel('Embedding Similarity')
    plt.ylabel('Frequency')
    plt.savefig(f"plots/{threshold}_embedding_histogram.png")

def similarity_heatmap(df):
    plt.figure()
    # plt.scatter(df['cache_llm_similarity'], df['cache_embedding_similarity'])
    plt.title('LLM Similarity vs Embedding Similarity')
    plt.xlabel('LLM Similarity')
    plt.ylabel('Embedding Similarity')
    # plt.figure(figsize=(10, 8))
    sns.kdeplot(
        x=df['true_llm_similarity'], 
        y=df['true_embedding_similarity'], 
        cmap="coolwarm",  # You can change the colormap
        fill=True,
        thresh=0,
        levels=100
    )
    plt.title("Heatmap for Similarities")
    plt.savefig(f"plots/{threshold}_heatmap.png")

thresholds = [5]
# overlapped_pdf(thresholds)
# similarity_vs_hitrate(thresholds)

for threshold in thresholds:
    print(f"Threshold: {threshold}")

    # Read TSV file
    l = ["image_id", "caption1", "caption2", "is_cache_hit", "hits_first_record", "cache_llm_similarity", "cache_embedding_similarity", "true_llm_similarity", "true_embedding_similarity", "latency"]
    df = pd.read_csv(f"embedding.tsv", sep='\t', names=l)

    pdf_plot(df)
    # histogram_plot(df)
    # similarity_heatmap(df)

    hit_counts = df['is_cache_hit'].astype(str).value_counts()
    total_queries = hit_counts.sum()
    hits = hit_counts.get('True', 0)
    hit_rate = hits / total_queries
    print(f"Cache Hit Rate: {hit_rate:.2%}")

    average_similarity = df['true_llm_similarity'].mean()
    print(f"Average LLM Similarity: {average_similarity:.4f}")

    average_similarity = df['true_embedding_similarity'].mean()
    print(f"Average Embedding Similarity: {average_similarity:.4f}")

    average_latency = df['latency'].mean()
    print(f"Average Latency: {average_latency:.4f}")
