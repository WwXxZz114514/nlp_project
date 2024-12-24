import matplotlib.pyplot as plt
import numpy as np

def visualize_score_distributions(file_path1, file_path2):
    def read_scores(file_path):
        scores = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    scores.append(int(line.strip()))
                except ValueError:
                    print(f"Warning: Skipping invalid line: {line.strip()} in {file_path}")
        return np.array(scores)
    
    scores1 = read_scores(file_path1)
    scores2 = read_scores(file_path2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(scores1, bins=10, alpha=0.7, label='Human NLEs Scores', color='skyblue', edgecolor='black')
    axs[1].hist(scores2, bins=10, alpha=0.7, label='X-ICL NLEs Scores', color='coral', edgecolor='black')

    axs[0].set_title('Distribution of Scores from Human NLEs')
    axs[1].set_title('Distribution of Scores from X-ICL NLEs')

    for ax in axs:
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.show()
    plt.savefig('score_distributions.png')

# Example usage
if __name__ == "__main__":
    dir_path = "./data/evaluation/"
    file1_path = "result_human.txt"
    file2_path = "result_xicl.txt"
    visualize_score_distributions(dir_path + file1_path, dir_path + file2_path)