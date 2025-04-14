import firebase_admin  # type: ignore[import-untyped]
from firebase_admin import auth  # noqa: F401

# TODO: SECURITY ISSUE - it's better to use .env variable for this.
# Instead, maybe use AWS secrets manager?
# cred_obj = firebase_admin.credentials.Certificate(
#     "/opt/app/secrets/vanna-portal-418018-1e27956b0918.json"
# )
cred_obj = firebase_admin.credentials.Certificate(
    {
        "type": "service_account",
        "project_id": "bitquant-afc7f",
        "private_key_id": "9ee0c58f90e51f9abbcbf848be89ae86d0ff4962",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQD8wIO3qYq/8GTf\njBifwcHIa/eZ4EVax8EXtGpzr0+rJmZOvfGWUsSDr0RoQCRgsoSwhyEUUJN39JLA\nAox/dpFmJKAvr6CM1PLzy/BW/sCnucSsfR4Skpcxf3TyA5pdeCD9zJuzDyIc7YeD\nMOV82+UG9ufq0BxuLMx96/SAgTBynRfPfqdS1K7A+Pz3avQYsCxk/aIUZmHR2G44\ngmyDltEK44/Dz2cNKAaLeoHq6141f4BH6YiYFpgibAmn0uQye0lGC9YeoTd7iHt3\nOOC0V59SKLuThCEFBnHLO0HKL5Hf5xcCHgyjMW/QIa4g9zso2wZ4NjwiyIOaQ9mZ\nSEP4EbunAgMBAAECggEAaG4KfioY3/EtXIf/7JIbxPmHFhbp0PZTu72Zdi7rFeYJ\nw2MLnHaaJ2aVNxW5chKQbHeInWIlbByrPZZQAqI0tSVQ4iMIjOe3ci/DH86mPyas\nMjCH6liTC6qD4TLH5vKpfvO5KSSJjbY+lV/wkcQxPs1pSUWvWX49B5IkNawrHk8j\nwgK+oocyxu3/6AclgoPYgn+3LlQEvXqCEHLYDnGNmUKWeJl4XMvbGDM26N0h2jff\nzPGGl7sdLRUCc08IaRBTmJwtVjYEzBoLLDEx+VdzCi+xUscuP+g1bgIAFb/jYnzF\nSPeuYHGYrCbyFbtWjmjR5CpkqpXuaSELdbBF6GJdAQKBgQD+lQxJrL5Z4F1B1xmu\n2n4GgNEkwYHFDXwb0zbzXOorwIg8WPlEj6EX44aWmae66MCmpj0Hb86iyBQ8PR8V\nSlFxvboU+CAoOqlU3pLNh89hIt4DJ+Tl5kGlK1/BkiKkPIC2f64s2yb0/ihihSjX\nKrglLDPE35XChf5hG1Cszm265wKBgQD+KNtzx0XfqdLSa8ojD1QFpjpaNeoCkwRY\nfnjeDX3/lshQ6QMqUN0PjlkWVUqSCujFDTFqZhOJMgJ+DjUpr1ZAWfonFOM9tcYk\nUza9tAQR7HtWnzyrxArN46YyhuYF6bzKTdzZXswg4Cie5Ljjm4QpHtDmNRZE0o0X\nAzxTe8ahQQKBgHg4IklYTbtbfC5vSS2M+B5SDnFw/7ryFz18jGJ36g0nKi51RaDe\nwo/pXdEYVmCpMPCBaChu4AF2wjeAFYGUSsmPcqQcV/MnYHc3c9Oi4odYU8bhu/Hk\nvfMlfF6Ih9tOxulnefSsuMTQkHmVsCeLgNHtAbVib/IgHHP21i4EfUTZAoGBAN4t\nRTm+t1xADmWXiIqBece+ekAl5Tz+28uoM2yZis2FN/NS3kt9iOFyZHpbcOad1sF7\nOPlz5hwGtZsQPHUGK3XxsfW8ErH9VwqmG7JVzUEF4wPkC5tzsqYtHToKJsaAf2Ky\nEh+K+RK8IYZVFzMQ3cU8hQzY13CuRlwZngC0sPyBAoGBAMsR/PMFAdBDscZB+4fA\ng0m9FMzQIlttPpNhDVYiEpnfkeJ6bU2T6kmIMRjz2OOMfdIChcgjXijtki6vmTqs\nwOikHHXbtUMSf5KYW4Mzf1/UKjpPAfxLuoRHuNUsGmVXyJg4gbx3jkDSa7mo/NDb\nI+cQp6X2K34NwplGeVfHRdBl\n-----END PRIVATE KEY-----\n",
        "client_email": "firebase-adminsdk-fbsvc@bitquant-afc7f.iam.gserviceaccount.com",
        "client_id": "105457998758090373807",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40bitquant-afc7f.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }
)


firebase = firebase_admin.initialize_app(cred_obj)
