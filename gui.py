import tkinter as tk
from tkinter import messagebox
import requests
from recommendation import generate_recommendations

def login_to_spotify():
    """Send a request to the Flask server for Spotify login."""
    try:
        response = requests.get("http://127.0.0.1:60000/login")
        if response.status_code == 200:
            messagebox.showinfo("Success", "Logged in to Spotify!")
        else:
            messagebox.showerror("Error", "Failed to log in.")
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", f"Could not connect to Flask server: {e}")

def handle_query():
    """Handles the user's query input and provides recommendations."""
    query = user_input.get()
    if not query.strip():
        messagebox.showwarning("Input Error", "Please enter a query!")
        return

    # Dummy response for testing purposes
    recommendations = ["Song 1", "Song 2", "Song 3"]
    result_text.set("\n".join(recommendations))

def build_gui():
    global user_input, result_text

    root = tk.Tk()
    root.title("RoastedRecs")
    root.geometry("600x500")

    login_button = tk.Button(root, text="Log in to Spotify", command=login_to_spotify)
    login_button.pack(pady=10)

    input_label = tk.Label(root, text="What are you looking for?")
    input_label.pack(pady=10)

    user_input = tk.Entry(root, width=50)
    user_input.pack(pady=10)

    query_button = tk.Button(root, text="Get Recommendations", command=handle_query)
    query_button.pack(pady=10)

    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text, justify="left", anchor="w")
    result_label.pack(pady=10, fill="both", expand=True)

    root.mainloop()
