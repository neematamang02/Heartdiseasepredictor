from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model

class SeparateAuthBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        if not username or not password:
            return None

        UserModel = get_user_model()

        try:
            user = UserModel.objects.get(username=username)
            if user.check_password(password):
                # For admin interface
                if request and request.path.startswith('/admin/'):
                    if user.is_staff:
                        return user
                    return None
                # For regular interface
                else:
                    if not user.is_staff:
                        return user
                    return None
            return None
        except UserModel.DoesNotExist:
            return None

    def get_user(self, user_id):
        UserModel = get_user_model()
        try:
            return UserModel.objects.get(pk=user_id)
        except UserModel.DoesNotExist:
            return None 