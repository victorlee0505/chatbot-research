# import json

# # Input JSON structure
# input_json = {
#     'functions': [{
#         'name': 'information_extraction',
#         'description': 'Extracts the relevant information from the passage.',
#         'parameters': {
#             'type': 'object',
#             'properties': {
#                 'first_name': {
#                     'title': 'First Name',
#                     'description': 'This is the first name of the user.',
#                     'type': 'string'
#                 },
#                 'last_name': {
#                     'title': 'Last Name',
#                     'description': 'This is the last name or surname of the user.',
#                     'type': 'string'
#                 },
#                 'full_name': {
#                     'title': 'Full Name',
#                     'description': 'Is the full name of the user ',
#                     'type': 'string'
#                 },
#                 'city': {
#                     'title': 'City',
#                     'description': 'The name of the city where someone lives',
#                     'type': 'string'
#                 },
#                 'email': {
#                     'title': 'Email',
#                     'description': 'an email address that the person associates as theirs',
#                     'type': 'string'
#                 },
#                 'language': {
#                     'title': 'Language',
#                     'enum': ['spanish', 'english', 'french', 'german', 'italian'],
#                     'type': 'string'
#                 }
#             },
#             'required': ['first_name', 'last_name', 'full_name', 'city', 'email', 'language']
#         }
#     }],
#     'function_call': {'name': 'information_extraction'}
# }

# # Transforming the structure
# transformed_json = {
#     'function_call': {
#         'name': input_json['function_call']['name'],
#         'arguments': json.dumps(
#             {k: "" for k in input_json['functions'][0]['parameters']['properties'].keys()},
#             indent=2
#         )
#     }
# }

# print(transformed_json)