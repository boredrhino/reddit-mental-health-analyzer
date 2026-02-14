"""
Reddit Mental Health Risk Analysis Tool
Written by: u/boredrhino
FOR RESEARCH PURPOSES ONLY - Use responsibly and ethically
Requires: praw, pandas, nltk, textblob, numpy
"""

import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import nltk
from textblob import TextBlob
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class RedditMentalHealthAnalyzer:
    """
    A tool for analyzing Reddit users' posts for potential mental health indicators.
    This is for research and early intervention purposes only.
    """
    
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize Reddit API connection
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Define keywords and patterns associated with mental health concerns
        self.risk_keywords = {
            'depression': [
                'depressed', 'hopeless', 'worthless', 'empty', 'numb',
                'can\'t go on', 'no reason to live', 'give up', 'tired of living'
            ],
            'anxiety': [
                'anxious', 'panic', 'overwhelmed', 'can\'t breathe',
                'heart racing', 'fear', 'dread', 'terrified'
            ],
            'suicidal': [
                'suicide', 'kill myself', 'end it all', 'better off dead',
                'want to die', 'end my life', 'suicidal thoughts'
            ],
            'self_harm': [
                'cut myself', 'self harm', 'hurt myself', 'burn myself',
                'self injury', 'cutting'
            ],
            'isolation': [
                'alone', 'no friends', 'nobody cares', 'isolated',
                'lonely', 'no one understands', 'left out'
            ],
            'hopelessness': [
                'no hope', 'never get better', 'pointless', 'useless',
                'waste', 'failure', 'nothing matters'
            ]
        }
        
        # Subreddits often containing mental health discussions
        self.mental_health_subs = [
            'depression', 'SuicideWatch', 'Anxiety', 'mentalhealth',
            'offmychest', 'TrueOffMyChest', 'selfharm', 'BPD',
            'bipolar', 'CPTSD', 'ptsd', 'socialanxiety'
        ]
        
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using VADER
        """
        if not text or len(text) < 10:
            return None
            
        try:
            sentiment = self.sia.polarity_scores(text)
            return sentiment
        except:
            return None
    
    def extract_risk_indicators(self, text):
        """
        Extract mental health risk indicators from text
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        risk_indicators = {}
        
        for category, keywords in self.risk_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(keyword)
            
            if matches:
                risk_indicators[category] = matches
        
        # Check for specific patterns (cries for help, urgency)
        urgent_patterns = [
            r'need (help|someone)',
            r'can\'t (take|handle) this',
            r'last (straw|hope)',
            r'give up',
            r'end it'
        ]
        
        urgent_matches = []
        for pattern in urgent_patterns:
            if re.search(pattern, text_lower):
                urgent_matches.append(pattern)
        
        if urgent_matches:
            risk_indicators['urgent_patterns'] = urgent_matches
        
        return risk_indicators
    
    def analyze_user(self, username, limit=50):
        """
        Analyze a single user's post history
        """
        try:
            user = self.reddit.redditor(username)
            user_data = {
                'username': username,
                'analysis_timestamp': datetime.now(),
                'total_posts': 0,
                'total_comments': 0,
                'avg_sentiment': None,
                'risk_indicators': Counter(),
                'recent_activity': [],
                'risk_score': 0,
                'warning_signs': []
            }
            
            posts_sentiment = []
            
            # Get user's recent posts and comments
            for item in user.new(limit=limit):
                try:
                    if hasattr(item, 'title') and hasattr(item, 'selftext'):
                        # It's a post
                        content = f"{item.title} {item.selftext}"
                        user_data['total_posts'] += 1
                    elif hasattr(item, 'body'):
                        # It's a comment
                        content = item.body
                        user_data['total_comments'] += 1
                    else:
                        continue
                    
                    # Analyze content
                    if len(content) > 10:  # Skip very short posts
                        sentiment = self.analyze_sentiment(content)
                        if sentiment:
                            posts_sentiment.append(sentiment['compound'])
                        
                        risk_indicators = self.extract_risk_indicators(content)
                        
                        if risk_indicators:
                            for category, matches in risk_indicators.items():
                                user_data['risk_indicators'][category] += len(matches)
                                
                                # Add to warning signs for high-risk categories
                                if category in ['suicidal', 'self_harm']:
                                    user_data['warning_signs'].append({
                                        'type': category,
                                        'matches': matches,
                                        'subreddit': item.subreddit.display_name if hasattr(item, 'subreddit') else 'unknown',
                                        'timestamp': datetime.fromtimestamp(item.created_utc)
                                    })
                    
                    # Store recent activity info
                    user_data['recent_activity'].append({
                        'subreddit': item.subreddit.display_name if hasattr(item, 'subreddit') else 'unknown',
                        'timestamp': datetime.fromtimestamp(item.created_utc)
                    })
                    
                except Exception as e:
                    continue
            
            # Calculate average sentiment
            if posts_sentiment:
                user_data['avg_sentiment'] = np.mean(posts_sentiment)
                user_data['sentiment_std'] = np.std(posts_sentiment)
            
            # Calculate risk score
            risk_score = 0
            
            # Factor 1: Sentiment (more negative = higher risk)
            if user_data['avg_sentiment']:
                if user_data['avg_sentiment'] < -0.5:
                    risk_score += 30
                elif user_data['avg_sentiment'] < -0.3:
                    risk_score += 20
                elif user_data['avg_sentiment'] < 0:
                    risk_score += 10
            
            # Factor 2: Risk indicators
            if 'suicidal' in user_data['risk_indicators']:
                risk_score += 50 * min(user_data['risk_indicators']['suicidal'], 3)
            if 'self_harm' in user_data['risk_indicators']:
                risk_score += 40 * min(user_data['risk_indicators']['self_harm'], 3)
            if 'hopelessness' in user_data['risk_indicators']:
                risk_score += 20 * min(user_data['risk_indicators']['hopelessness'], 3)
            
            # Factor 3: Activity in mental health subreddits
            mh_activity = sum(1 for act in user_data['recent_activity'] 
                             if act['subreddit'].lower() in self.mental_health_subs)
            
            if mh_activity > 10:
                risk_score += 20
            elif mh_activity > 5:
                risk_score += 10
            
            user_data['risk_score'] = min(risk_score, 100)  # Cap at 100
            
            return user_data
            
        except Exception as e:
            print(f"Error analyzing user {username}: {str(e)}")
            return None
    
    def find_high_risk_users(self, subreddits=None, limit=100, post_limit=30):
        """
        Find potentially high-risk users by scanning subreddits
        """
        if subreddits is None:
            subreddits = self.mental_health_subs
        
        high_risk_users = []
        analyzed_users = set()
        
        for subreddit_name in subreddits:
            print(f"Scanning r/{subreddit_name}...")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get recent posts from the subreddit
                for post in subreddit.new(limit=limit):
                    try:
                        # Analyze post author
                        if post.author and post.author.name not in analyzed_users:
                            user_data = self.analyze_user(post.author.name, limit=post_limit)
                            
                            if user_data and user_data['risk_score'] > 50:  # High-risk threshold
                                high_risk_users.append(user_data)
                                analyzed_users.add(post.author.name)
                                
                                print(f"  Found high-risk user: {post.author.name} (Score: {user_data['risk_score']})")
                            
                            # Rate limiting
                            time.sleep(0.5)
                    
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Error scanning {subreddit_name}: {str(e)}")
                continue
        
        return high_risk_users
    
    def generate_report(self, users_data, output_file='mental_health_risk_report.csv'):
        """
        Generate a report of analyzed users
        """
        if not users_data:
            print("No data to generate report")
            return
        
        report_data = []
        
        for user in users_data:
            # Prepare warning signs summary
            warning_summary = '; '.join([f"{w['type']}: {', '.join(w['matches'])}" 
                                        for w in user['warning_signs'][:3]])
            
            report_data.append({
                'username': user['username'],
                'risk_score': user['risk_score'],
                'avg_sentiment': user['avg_sentiment'],
                'total_posts': user['total_posts'],
                'total_comments': user['total_comments'],
                'risk_indicators': dict(user['risk_indicators']),
                'warning_signs': warning_summary,
                'analysis_date': user['analysis_timestamp']
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(report_data)
        df = df.sort_values('risk_score', ascending=False)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nReport saved to {output_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MENTAL HEALTH RISK ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total users analyzed: {len(users_data)}")
        print(f"High-risk users (score > 70): {len(df[df['risk_score'] > 70])}")
        print(f"Medium-risk users (score 50-70): {len(df[(df['risk_score'] > 50) & (df['risk_score'] <= 70)])}")
        
        return df

def main():
    """
    Main function to run the analyzer
    """
    # Configuration
    CLIENT_ID = "YOUR_CLIENT_ID"
    CLIENT_SECRET = "YOUR_CLIENT_SECRET"
    USER_AGENT = "MentalHealthResearch/1.0 (by /u/YOUR_USERNAME)"
    
    # Initialize analyzer
    analyzer = RedditMentalHealthAnalyzer(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    
    print("Reddit Mental Health Risk Analyzer")
    print("=" * 40)
    print("\nNOTE: This tool is for research purposes only.")
    print("Always prioritize user privacy and ethical considerations.")
    print("If someone appears to be in crisis, direct them to professional help.\n")
    
    # Find high-risk users
    print("Scanning for high-risk users...")
    high_risk_users = analyzer.find_high_risk_users(
        subreddits=['SuicideWatch', 'depression', 'mentalhealth'],
        limit=50,
        post_limit=20
    )
    
    # Generate report
    if high_risk_users:
        report_df = analyzer.generate_report(high_risk_users)
        
        # Print top 10 highest risk users
        print("\nTop 10 Highest Risk Users:")
        print(report_df[['username', 'risk_score', 'avg_sentiment']].head(10))
        
        # Save detailed report
        report_df.to_csv('detailed_mental_health_report.csv', index=False)
    else:
        print("No high-risk users found in this scan.")

if __name__ == "__main__":
    main()
