# Google Drive API Setup Guide

## Overview
To access Google Docs (.gdoc files) from your Google Drive, you need to set up Google Drive API credentials.

## Step 1: Install Required Packages
```bash
pip install langchain-google-community[drive]
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

## Step 2: Set Up Google Cloud Project

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Create or select a project**:
   - Click "Select a project" → "New Project"
   - Give it a name like "RAG Document Loader"
   - Click "Create"

3. **Enable Google Drive API**:
   - Go to "APIs & Services" → "Library"
   - Search for "Google Drive API"
   - Click on it and click "Enable"

## Step 3: Create Credentials

1. **Go to Credentials**:
   - Navigate to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"

2. **Configure OAuth consent screen** (if first time):
   - Click "Configure Consent Screen"
   - Choose "External" user type
   - Fill in required fields:
     - App name: "RAG Document Loader"
     - User support email: your email
     - Developer contact: your email
   - Click "Save and Continue"
   - Skip Scopes section (click "Save and Continue")
   - Add your email as a test user
   - Click "Save and Continue"

3. **Create OAuth Client ID**:
   - Application type: "Desktop application"
   - Name: "RAG Desktop Client"
   - Click "Create"

4. **Download credentials**:
   - Click "Download JSON"
   - Save as `credentials.json` in your project directory

## Step 4: Get Your Folder ID

1. Open your Google Drive folder in a web browser
2. Copy the folder ID from the URL:
   ```
   https://drive.google.com/drive/folders/1ABC123XYZ456
                                        ↑ This is your folder ID
   ```

## Step 5: Update Your Script

Replace `YOUR_FOLDER_ID_HERE` in `LoadDocumentWithGoogleDrive.py` with your actual folder ID.

## Step 6: First Run Authentication

When you first run the script:
1. It will open a browser window
2. Sign in with your Google account
3. Grant permissions to access Google Drive
4. A `token.json` file will be created automatically
5. Future runs won't require browser authentication

## Troubleshooting

### "Access blocked" error
- Make sure your app is in "Testing" mode in OAuth consent screen
- Add your email as a test user

### "Quota exceeded" error
- Google Drive API has usage limits
- Wait a few minutes and try again
- Consider using pagination for large folders

### Files not loading
- Check folder permissions (make sure your account has access)
- Verify the folder ID is correct
- Some files might be in subfolders (use `recursive=True`)

## Security Note

- Keep `credentials.json` and `token.json` secure
- Don't commit them to version control
- Add them to your `.gitignore` file

## Alternative: Manual Export

If API setup is too complex, you can manually export Google Docs:
1. Open each Google Doc
2. File → Download → Microsoft Word (.docx)
3. Save to your local folder
4. Your existing script will process the .docx files 