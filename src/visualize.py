import matplotlib.pyplot as plt
import sys
import json

# ROUGE scores generated from running main.py
SCORES_FILE: str = "scores.json"


def visualize(metric="1", path=SCORES_FILE):
    """ Visualize evaluation scores with a bar graph. "metric" is a string
    representing which ROUGE score to visualize (i.e. "1", "2", or "l") """
    with open(path, 'r', encoding='utf8') as f:
        json_scores = json.load(f)

    x_vals = [i for i in range(len(json_scores))]
    r_vals = [score[0][f"rouge-{metric}"]["r"] for score in json_scores]
    p_vals = [score[0][f"rouge-{metric}"]["p"] for score in json_scores]
    f_vals = [score[0][f"rouge-{metric}"]["f"] for score in json_scores]

    barWidth = .25
    fig = plt.subplots(figsize=(12, 8))

    br1 = x_vals
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, r_vals, width=barWidth, label="Recall")
    plt.bar(br2, p_vals, width=barWidth, label="Precision")
    plt.bar(br3, f_vals, width=barWidth, label="F-1")
    plt.xlabel("Document")
    plt.ylabel("Score")
    plt.title(f"Rouge-{metric}")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    visualize(sys.argv[1])
