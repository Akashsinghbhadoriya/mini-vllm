def client(server, prompt):

    response = server.submit_request(prompt)

    print("--------------------------------")

    print("Prompt:")

    print(prompt)

    print()

    print("Response:")

    print(response)

    print("--------------------------------")