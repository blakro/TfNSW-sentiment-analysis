import argparse
import pandas as pd
import torch
from facebook_scraper import get_posts
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedFacebookScraper:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the scraper and load the sentiment analysis model.
        """
        logging.info(f"Loading sentiment model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logging.info(f"Model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a given text using RoBERTa.
        Returns a dictionary with label and score (prob_pos - prob_neg).
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {'sentiment_label': 'neutral', 'sentiment_score': 0.0}

        # Truncate text to max length allowed by the model
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded_input)

        scores = output.logits[0].cpu().detach().numpy()
        scores = softmax(scores)

        # Mapping for cardiffnlp/twitter-roberta-base-sentiment-latest
        # 0 -> negative, 1 -> neutral, 2 -> positive
        # Check config to be sure, but this is standard for this model

        labels = self.config.id2label
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        top_label = labels[ranking[0]]

        # Calculate a compound-like score: Positive Prob - Negative Prob
        # Assuming index 0 is Negative and index 2 is Positive (verify with specific model config if needed)
        # For twitter-roberta-base-sentiment: 0: negative, 1: neutral, 2: positive

        neg_score = scores[0]
        neu_score = scores[1]
        pos_score = scores[2]

        compound_score = pos_score - neg_score

        return {
            'sentiment_label': top_label,
            'sentiment_score': compound_score,
            'prob_neg': neg_score,
            'prob_neu': neu_score,
            'prob_pos': pos_score
        }

    def scrape(self, target_id, pages=1, cookies=None, options=None):
        """
        Scrape posts from a Facebook Page or Group.
        """
        posts_data = []

        scrape_kwargs = {
            "group": target_id, # This is just a label for the first arg, get_posts handles page vs group often automatically or via args
            "pages": pages,
            "options": options or {"comments": False} # Default to no comments for speed unless requested
        }

        # Handling Group vs Page logic implied by cookies
        # facebook_scraper's get_posts first argument is 'account' (page name or group id)
        # It detects groups if the ID looks like a number, but explicit cookies help.

        if cookies:
            scrape_kwargs["cookies"] = cookies

        logging.info(f"Starting scrape for target: {target_id}, Pages: {pages}, Cookies provided: {bool(cookies)}")

        try:
            # get_posts is a generator
            for post in get_posts(target_id, pages=pages, cookies=cookies, options=scrape_kwargs["options"]):

                # Extract fields
                text = post.get('text', '')

                # Perform sentiment analysis
                sentiment_result = self.analyze_sentiment(text)

                post_entry = {
                    'post_id': post.get('post_id'),
                    'time': post.get('time'),
                    'text': text,
                    'likes': post.get('likes'),
                    'comments': post.get('comments'),
                    'shares': post.get('shares'),
                    'sentiment': sentiment_result['sentiment_score'],
                    'sentiment_label': sentiment_result['sentiment_label']
                }

                posts_data.append(post_entry)
                logging.info(f"Scraped post {post.get('post_id')} | Sentiment: {sentiment_result['sentiment_label']} ({sentiment_result['sentiment_score']:.2f})")

        except Exception as e:
            logging.error(f"Error during scraping: {e}")
            # If no cookies provided and it failed, hint about cookies
            if not cookies and "Login required" in str(e):
                logging.warning("This target might be a Group or age-restricted Page requiring login. Please provide cookies.")

        return pd.DataFrame(posts_data)

def main():
    parser = argparse.ArgumentParser(description="Advanced Facebook Scraper with RoBERTa Sentiment Analysis")
    parser.add_argument("target", help="Facebook Page Name or Group ID")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to scrape")
    parser.add_argument("--cookies", type=str, help="Path to cookies.txt or cookies.json file (Required for Groups)")
    parser.add_argument("--output", type=str, default="fb_scrape_results.csv", help="Output CSV filename")
    parser.add_argument("--comments", action="store_true", help="Scrape comments (slower)")

    args = parser.parse_args()

    scraper = AdvancedFacebookScraper()

    options = {"comments": args.comments, "progress": True}

    df = scraper.scrape(args.target, pages=args.pages, cookies=args.cookies, options=options)

    if not df.empty:
        # Add a simple ID column (index)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'ID'}, inplace=True)

        logging.info(f"Scraping complete. Collected {len(df)} posts.")
        df.to_csv(args.output, index=False)
        logging.info(f"Results saved to {args.output}")
    else:
        logging.warning("No posts collected.")

if __name__ == "__main__":
    main()
