from flask import Flask, render_template, request
from model import recommend_with_sentiment
import pandas as pd
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    username = request.form["username"]
    df=pd.read_csv('sample30.csv')
    try:
        products = recommend_with_sentiment(username, df)

        if products.empty:
            return render_template(
                "index.html",
                message="No recommendations found"
            )

        # Convert dataframe to list of dicts for HTML
        products = products.to_dict(orient="records")

        return render_template("index.html", products=products)

    except Exception as e:
        print("ERROR:", e)
        return render_template("index.html", message=str(e))



if __name__ == "__main__":
    app.run(debug=True)
