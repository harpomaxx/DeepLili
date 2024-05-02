import dropbox
import os
import json
import logging

TOKEN_STORE_PATH = 'dropbox_tokens.json'

logging.basicConfig(level=logging.INFO, filename='dropbox_upload.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def read_config(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load JSON data from the file
        config = json.load(file)
    
    # Extract the variables
    app_key = config.get('APP_KEY')
    app_secret = config.get('APP_SECRET')
    
    # Return the variables
    return app_key, app_secret

def load_tokens():
    if os.path.exists(TOKEN_STORE_PATH):
        with open(TOKEN_STORE_PATH, 'r') as token_file:
            tokens = json.load(token_file)
            logging.info("Loading tokens successfully.")
            return tokens['access_token'], tokens['refresh_token']
    else:
            logging.info("Could not find tokens file.")

    return None, None

def save_tokens(access_token, refresh_token):
    tokens = {'access_token': access_token, 'refresh_token': refresh_token}
    with open(TOKEN_STORE_PATH, 'w') as token_file:
        json.dump(tokens, token_file)
    logging.info("Tokens saved. Access token refreshed.")

def create_dropbox_client(refresh_token):
    dbx = dropbox.Dropbox(oauth2_refresh_token=refresh_token, app_key=APP_KEY, app_secret=APP_SECRET)
    return dbx

def upload_file(file_path, target_path):
    access_token, refresh_token = load_tokens()
    if not access_token or not refresh_token:
        logging.error("Tokens not available.")
        return

    dbx = create_dropbox_client(refresh_token)
    try:
        with open(file_path, "rb") as file:
            dbx.files_upload(file.read(), target_path, mode=dropbox.files.WriteMode.overwrite)
            logging.info("File uploaded successfully.")
            # Save the potentially updated access token
            save_tokens(dbx._oauth2_access_token, refresh_token)
    except dropbox.exceptions.AuthError as e:
        logging.error("Authentication error: %s", e)

## Usage
APP_KEY, APP_SECRET = read_config('config.json')
#print("APP_KEY:", app_key)
#print("APP_SECRET:", app_secret)

