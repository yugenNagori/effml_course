import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression


def main():
    fig, ax = plt.subplots()
    
    path = "results/reference/grid_reduced.json"
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    groups = {}
    for elem in data['capacity']:
        groups.setdefault((elem["arm"], elem["seed"]), []).append((elem['d_k'], elem['m_star']))
    
    colors = {("linear", 0): "tab:blue", ("linear", 1): "tab:orange",
                     ("delta", 0): "tab:green", ("delta", 1): "tab:red"}
    
    for (arm, seed), points in groups.items():
        label = f"{arm} seed={seed}"
        color = colors[(arm, seed)]
        X = [[dk] for dk, m_star in points]
        y = [m_star for dk, m_star in points]
        if (arm, seed) == ("linear", 0):
            ax.scatter(X, y, label=label, facecolors="none", edgecolors=color,
                       s=120, linewidths=2, zorder=7)
        else:
            ax.scatter(X, y, label=label, color=color, s=48, zorder=3)
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        print(arm, seed, "c =", model.coef_[0], "b =", model.intercept_)
        ax.plot(X, model.predict(X), color=color, label=f"{label} МНК")
        
        
    print("Точки с запуска")
    with open("results/grid_student.json", encoding="utf-8") as f:
        student = json.load(f)

    student_groups = {}
    for elem in student["capacity"]:
        if elem["censored"] == "below":
            ax.scatter(elem["d_k"], 8, marker="v",
                    facecolors="none", edgecolors="black")
        elif elem["censored"] == "above":
            ax.scatter(elem["d_k"], 32, marker="^",
                    facecolors="none", edgecolors="black")
        else:
            student_groups.setdefault((elem["arm"], elem["seed"]), []).append((elem["d_k"], elem["m_star"]))

    for (arm, seed), points in student_groups.items():
        label = f"{arm} seed={seed}"
        color = "tab:purple" if seed == 331181 else "tab:brown"
        X = [[dk] for dk, m_star in points]
        y = [m_star for dk, m_star in points]
        ax.scatter(X, y, label=label, marker="D", facecolors="none",
                   edgecolors=color, s=90, linewidths=1.8, zorder=5)
        if len(points) < 2:
            print(arm, seed, "прямую построить нельзя")
            continue
        model = LinearRegression(fit_intercept=True).fit(X, y)
        print(arm, seed, "c =", model.coef_[0], "b =", model.intercept_)
        ax.plot(X, model.predict(X), color=color, label=f"{label} МНК")

    ax.plot([8, 16], [4.5, 12.5], "k--", label="гипотеза: M* = d_k - 3.5")
    ax.set_xlabel("d_k")
    ax.set_ylabel("M*")
    ax.set_ylim(0, 34)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.savefig("figures/collapse-vs-dk.png", bbox_inches="tight", dpi=160)
    
if __name__ == '__main__':
    main()
