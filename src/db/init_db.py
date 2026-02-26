import sqlite3

def init_db():
    conn = sqlite3.connect('chainlit.db')
    cursor = conn.cursor()
    
    # 1. USERS
    cursor.execute('CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, identifier TEXT NOT NULL UNIQUE, metadata TEXT NOT NULL, createdAt TEXT)')
    
    # 2. THREADS
    cursor.execute('CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, createdAt TEXT, name TEXT, userId TEXT, userIdentifier TEXT, metadata TEXT, tags TEXT, FOREIGN KEY (userId) REFERENCES users(id))')

    # 3. STEPS
    cursor.execute('''CREATE TABLE IF NOT EXISTS steps (
        id TEXT PRIMARY KEY, name TEXT, type TEXT, threadId TEXT, parentId TEXT, streaming BOOLEAN, 
        waitForAnswer BOOLEAN DEFAULT 0, isError BOOLEAN DEFAULT 0, metadata TEXT, tags TEXT, 
        input TEXT, output TEXT, createdAt TEXT, start TEXT, end TEXT, generation TEXT, 
        showInput TEXT, language TEXT, indent INTEGER, defaultOpen BOOLEAN DEFAULT 0, feedback TEXT,
        FOREIGN KEY (threadId) REFERENCES threads(id))''')

    # 4. ELEMENTS (New table required for resume)
    cursor.execute('''CREATE TABLE IF NOT EXISTS elements (
        id TEXT PRIMARY KEY, threadId TEXT, type TEXT, chainlitKey TEXT, url TEXT, 
        objectKey TEXT, name TEXT, display TEXT, size TEXT, language TEXT, 
        page INTEGER, forId TEXT, mime TEXT, props TEXT,
        FOREIGN KEY (threadId) REFERENCES threads(id))''')

    # 5. FEEDBACKS (New table required for resume)
    cursor.execute('''CREATE TABLE IF NOT EXISTS feedbacks (
        id TEXT PRIMARY KEY, forId TEXT, value INTEGER, comment TEXT,
        FOREIGN KEY (forId) REFERENCES steps(id))''')
    
    conn.commit()
    conn.close()
    print("Database fully initialized with all required tables!")

if __name__ == "__main__":
    init_db()