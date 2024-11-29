from transformers import pipeline

def generate_sarcasm(user_data, recommendations, query):
    """
    Generates a sarcastic response using the SarcasMLL-1B model based on user profile and recommendations.
    """
    # Use Hugging Face Transformers pipeline with the SarcasMLL-1B model
    generator = pipeline("text-generation", model="AlexandrosChariton/SarcasMLL-1B", device=0, truncation=False)

    # Refine the prompt to focus on sarcasm
    profile_summary = f"Top artist: {user_data['top_tracks'][0]['artist']}, Average energy: {user_data['audio_features']['energy'].mean():.2f}"
    prompt = (
        f"User listens to {profile_summary}.\n"
        f"Recommendations: {', '.join(recommendations[:3])}.\n"
        f"Generate a short sarcastic remark about their taste in music."
    )

    # generate the sarcastic response:
    try:
        response = generator(prompt, max_new_tokens=100, num_return_sequences=1, pad_token_id=50256)
        generated_text = response[0]["generated_text"].strip()

        # Remove the prompt from the generated text
        if generated_text.startswith(prompt):
            sarcasm_only = generated_text[len(prompt):].strip()
        else:
            sarcasm_only = generated_text

        # Truncate after the last period
        clean_response = truncate_after_last_period(sarcasm_only)
        return clean_response
    except Exception as e:
        # Handle model errors gracefully
        return f"Oops, something went wrong while generating sarcasm: {str(e)}"

def truncate_after_last_period(response):
    """
    Truncates the response string after the last period ('.').
    If no period is found, returns the full response.
    """
    if '.' in response:
        last_period_index = response.rfind('.')
        return response[:last_period_index + 1]  # Include the last period
    return response.strip()  # Return the full response if no period is found

