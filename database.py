import hashlib

# Dummy in-memory database (Replace with a real DB in production)
users_db = {}

# Function to hash the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Check if the username already exists
def is_username_taken(username):
    return username in users_db

# Check if the email already exists
def is_email_taken(email):
    for user in users_db.values():
        if user['email'] == email:
            return True
    return False

# Get user from DB (dummy function)
def get_user_from_db(username):
    return users_db.get(username)

# Create a new user in DB
def create_user_in_db(username, email, password):
    hashed_password = hash_password(password)
    users_db[username] = {'username': username, 'email': email, 'password': hashed_password}
