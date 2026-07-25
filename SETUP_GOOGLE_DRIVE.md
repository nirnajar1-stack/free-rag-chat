# חיבור Google Drive בענן (בלי להתקין Google Drive על המחשב)

המערכת מתחברת לתיקייה ב-Google Drive דרך **Service Account** של Google Cloud.
כך אפשר:
1. להעלות קבצים מהאפליקציה **ישירות לדרייב בענן**
2. לשים קבצים בתיקייה בדרייב (בדפדפן) — והאפליקציה **תסנכרן ותאנדקס אוטומטית**

---

## שלב 1 — יצירת פרויקט ו-Service Account

1. היכנס ל-[Google Cloud Console](https://console.cloud.google.com/)
2. צור פרויקט חדש (או בחר קיים)
3. בתפריט: **APIs & Services → Library**
4. חפש **Google Drive API** → Enable
5. עבור ל-**APIs & Services → Credentials → Create Credentials → Service account**
6. תן שם (למשל `rag-drive-sync`) → Create → Done
7. לחץ על ה-Service Account שנוצר → לשונית **Keys**
8. **Add key → Create new key → JSON** → הורד את הקובץ
9. שמור את הקובץ בפרויקט בשם, למשל: `secrets/gcp-service-account.json`  
   (אל תעלה אותו ל-Git)

העתק מה-JSON את השדה `client_email` (נראה כמו `rag-drive-sync@....iam.gserviceaccount.com`).

---

## שלב 2 — תיקייה ב-Google Drive

1. ב-[Google Drive](https://drive.google.com) צור תיקייה, למשל `RAG_Docs`
2. פתח את התיקייה → לחץ ימין → **Share / שיתוף**
3. הדבק את ה-`client_email` של ה-Service Account
4. תן הרשאת **Editor / עורך** → Send
5. העתק את **Folder ID** מהכתובת בדפדפן:

```
https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX
                                         ^^^^^^^^^^^^^^^^^^^^
                                         זה ה-Folder ID
```

---

## שלב 3 — הגדרה מקומית (`.env`)

```env
GOOGLE_DRIVE_FOLDER_ID=XXXXXXXXXXXXXXXXXXXX
GOOGLE_APPLICATION_CREDENTIALS=secrets/gcp-service-account.json
```

או להדביק את כל ה-JSON:

```env
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
```

---

## שלב 4 — הגדרה ב-Streamlit Cloud (Secrets)

ב-App settings → Secrets הדבק לדוגמה:

```toml
GROQ_API_KEY = "gsk_..."
GOOGLE_DRIVE_FOLDER_ID = "XXXXXXXXXXXXXXXXXXXX"

# אפשר כמחרוזת JSON בשורה אחת:
GOOGLE_SERVICE_ACCOUNT_JSON = """
{
  "type": "service_account",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "...@....iam.gserviceaccount.com",
  "client_id": "...",
  "token_uri": "https://oauth2.googleapis.com/token"
}
"""
```

---

## שימוש

### העלאה מהאפליקציה לדרייב
בסרגל הצד → העלאת מסמכים → הקבצים עולים **לתיקייה בדרייב בענן** ואז נבנה אינדקס.

### העלאה ידנית בדרייב
שים PDF/TXT/MD בתיקייה המשותפת ב-drive.google.com.  
בפתיחת האפליקציה (או בלחיצה על **סנכרן מ-Drive**) המערכת מורידה קבצים חדשים/מעודכנים ובונה אינדקס אוטומטית.

---

## פתרון תקלות

| בעיה | פתרון |
|------|--------|
| `File not found` / ריק | ודא ששיתפת את התיקייה עם ה-`client_email` |
| `API not enabled` | הפעל Google Drive API בפרויקט |
| אין שינוי אחרי העלאה בדרייב | לחץ **סנכרן מ-Drive עכשיו** או רענן את האפליקציה |
