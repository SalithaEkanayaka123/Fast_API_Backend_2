from db.crud import get_all_users
def get_user_list():
    users_list = []
    users = get_all_users()
    users = [[x.serialize for x in users.all()]]
    print (users)
