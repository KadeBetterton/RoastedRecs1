# Harmonaic 🎵  
**A Personalized Music Recommendation Assistant**

Harmonaic is a music recommendation system that uses **machine learning and natural language processing (NLP)** to suggest songs based on user input. The system leverages clustering, content-based filtering, and an interactive chatbot to provide tailored music recommendations.
_________________________________________________________________________________________________________

To use the recommendation system, download harmoniac.py and english_songs.csv to the same location, and run harmoniac.py (More detailed guidance below).
The Walthrough zip file contains a Jupyter Notebook file of data visualizations.
_________________________________________________________________________________________________________

## Features  
- **Artist Name Extraction**: Uses an **LLM (Gemini API)** to extract and correct artist names from user input.  
- **Song Recommendations**: Generates music recommendations based on:  
  - K-Means clustering (unsupervised learning).  
  - Content-based similarity (cosine similarity of song features).  
- **Data Processing & Feature Engineering**:  
  - Preprocesses song metadata (removing duplicates, filtering out certain genres).  
  - Scales features using **MinMaxScaler** and applies **PCA** for dimensionality reduction.  
- **Chatbot Interface**:  
  - Built with **Tkinter** for interactive user engagement.  
  - Allows users to explore new music through conversational interactions.  
- **Evaluation Metrics**:  
  - **Mean Reciprocal Rank (MRR)** for ranking accuracy.  
  - **Diversity Score** for ensuring variety in recommendations.  

---

## 🛠 Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/raquelanamb/harmonaic.git
   cd harmonic

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your Google Gemini API Key (for artist name extraction & explanations), and replace it in this line in Harmoniac.py:
```
genai.configure(api_key="your api key")
```

## 📖 Usage

Run the application with:
```
python main.py
```


## How it Works

1. Enter an artist name (e.g., “Beyoncé”).
2. Harmonaic fetches a list of songs by the artist.
3. Select a song from the numbered list.
4. Choose a recommendation method:
   (1) K-Means Clustering
   (2) Content-Based Filtering
5. Get personalized song recommendations and an explanation of why they match your choice.


## Data

Dataset used: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

This is a CSV dataset of songs, including features like:
- danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- Genres (with some excluded, e.g., anime, kids, sleep, etc.)


## Technologies Used

- Python
- Tkinter (GUI)
- Pandas
- NumPy (Data processing)
- Scikit-learn (Clustering, similarity, PCA)
- Google Gemini API (LLM-powered artist extraction & explanations)
- Matplotlib (Data visualization)


## Example Output

Welcome! I'm here to help you find new music similar to what you already enjoy.
Tell me an artist you like, or you can type 'quit' to exit.

You: i want to find songs like brandi carlile's

Okay, here are some songs by Brandi Carlile:
1. Party of One
2. Throwing Good After Bad
3. This Time Tomorrow
4. When You're Wrong
5. You and Me on the Rock

Enter the number of a song for recommendations:

You: 2

Okay, I'll recommend you songs musically similar to "Throwing Good After Bad."

Please choose a recommendation method (1 for K-means, 2 for Content-based):

You: 2

My recommendations:
- cold / mess - Audiotree Live Version by Prateek Kuhad (folk)
- Pervyy sneg by Gelena Velikanova (romance)
- When God Dips His Love In My Heart - The March Of Dimes Show (Single Version) by Hank Williams (honky-tonk)
- Helplessly Hoping - 2005 Remaster by Crosby, Stills & Nash (blues)
- The Christmas Song by Aloe Blacc (soul)

Why you'll like these:
"Throwing Good After Bad" by Brandi Carlile evokes a powerful blend of vulnerability and resilience, often characterized by a strong,
emotive vocal delivery, layered instrumentation, and a folk-pop sensibility. "cold / mess" by Prateek Kuhad shares the raw emotional
core, with a similar focus on intimate storytelling and melancholic introspection, suggestive of Russian folk music. The "When God 
Dips His Love In My Heart" by Hank Williams offers a different yet comparable raw emotional honesty, though firmly rooted in country 
music tradition. The stripped-down, acoustic approach of Crosby, Stills & Nash's "Helplessly Hoping" finds resonance in its reflective 
mood, intimate vocal harmonies; the nostalgic mood and thoughtful lvrical content of these songs link them to the introspective 
undercurrents of Carlile's work. Finally, Aloe Blacc's "The Christmas Song" demonstrates a different yet relevant style. While a 
holiday song, its soft and mellow approach could appeal to listeners drawn to the quiet emotional strength and folk foundations that 
often feature in Carlile's music.

The MMR score of 1.00 means these songs are very musically similar to "Throwing Good After Bad."
The diversity of these songs is 0.00, meaning they are very similar to each other.

Would you like to start with a new artist or quit? Type 'new artist' or 'quit'.
