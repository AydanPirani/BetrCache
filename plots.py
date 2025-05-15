import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick
from scipy.ndimage import gaussian_filter1d

def get_embedding_similarity(vals, output_file):
    plt.figure(figsize=(6,4))
    
    ax = sns.kdeplot( data=vals, fill=True)

    ax.set_title('Embedding Similarity Distribution')
    ax.set_xlabel('Embedding Similarity')
    ax.set_ylabel('Density')
    
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file)

def get_llm_similarity(vals, output_file):
    plt.figure(figsize=(6,4))
    
    ax = sns.kdeplot(data=vals, fill=True)

    ax.set_title('LLM Similarity Distribution')
    ax.set_xlabel('LLM Similarity')
    ax.set_ylabel('Density')

    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file)


def get_scatter(data, output_file):
    plt.figure(figsize=(6,4))
    
    # ax = sns.scatterplot(data=data, x='llm_similarity', y='embedding_similarity')
    ax = sns.kdeplot(
        x=data['llm_similarity'].dropna(),
        y=data['embedding_similarity'].dropna(),
        thresh=0,       # show full range
        levels=100,     # smooth gradations
        cmap="mako"     # default colormap (you can omit this line for seaborn’s default)
    )

    ax.set_title('LLM Similarity vs Embedding Similarity')
    ax.set_xlabel('LLM Similarity')
    ax.set_ylabel('Embedding Similarity')

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file)

def gen_plots(threshold):
    print("Generating plots for threshold: ", threshold)
    # Read TSV file
    l = ["image_id", "caption1", "caption2", "is_cache_hit", "hits_first_record", "llm_similarity", "embedding_similarity", "act_llm_similarity", "act_embedding_similarity", "latency"]
    df = pd.read_csv(f"outputs/embedding_{threshold}.tsv", sep='\t', names=l)

    # LLM Similarity
    llm_output_file = f"plots/{threshold}_llm_similarity_kde.png"
    data = df['llm_similarity'].dropna()
    # get_llm_similarity(data, llm_output_file)

    # Embedding Similarity
    embedding_output_file = f"plots/{threshold}_embedding_similarity_kde.png"
    data = df['embedding_similarity'].dropna()
    # get_embedding_similarity(data, embedding_output_file)
    
    # Scatter plot: LLM vs Embedding Similarity
    scatter_output_file = f"plots/{threshold}_scatter.png"
    data = df[['llm_similarity', 'embedding_similarity']].dropna()
    get_scatter(data, scatter_output_file)

    # # Bar chart: Cache Hit vs Miss Counts
    # hit_counts = df['is_cache_hit'].astype(str).value_counts()
    # total_queries = hit_counts.sum()
    # hits = hit_counts.get('True', 0)
    # hit_rate = hits / total_queries
    # print(f"Cache Hit Rate: {hit_rate:.2%}")

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

    # for label, group in zip(labels, groups):
    #     q1 = np.percentile(group, 25)
    #     median = np.median(group)
    #     q3 = np.percentile(group, 75)
    #     iqr = q3 - q1
    #     lower_whisker = max(min(group), q1 - 1.5 * iqr)
    #     upper_whisker = min(max(group), q3 + 1.5 * iqr)
        
    #     print(f"Group: {label}")
    #     print(f"Min: {lower_whisker:.4f}, Q1: {q1:.4f}, Median: {median:.4f}, Q3: {q3:.4f}, Max: {upper_whisker:.4f}\n")


if __name__ == "__main__":
    gen_plots(0.6)
    gen_plots(0.7)
    gen_plots(0.8)
    gen_plots(0.9)
