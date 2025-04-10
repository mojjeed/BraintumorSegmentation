import gdown
import os

def download_model_if_needed():
    model_path = 'models/model.h5'
    if not os.path.exists(model_path):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Replace this with your Google Drive shared link
        url = 'YOUR_GOOGLE_DRIVE_SHARE_LINK'
        
        # Convert to direct download link
        file_id = url.split('/')[-2]
        direct_url = f'https://drive.google.com/uc?id={file_id}'
        
        # Download the file
        gdown.download(direct_url, model_path, quiet=False)
        st.success("Model downloaded successfully!")

if 'model_downloaded' not in st.session_state:
    st.session_state.model_downloaded = False

if not st.session_state.model_downloaded:
    download_model_if_needed()
    st.session_state.model_downloaded = True

# ... rest of your code ... 