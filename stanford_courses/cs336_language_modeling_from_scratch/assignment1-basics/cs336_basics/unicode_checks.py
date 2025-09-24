test_string = "hello! こんにちは!"
print(f"\nInitial string: {test_string}")
print(len(test_string))

utf8_encoded = test_string.encode("utf-8")
print(f"\nBytes -- utf-8 encoded: {utf8_encoded}")
print(len(utf8_encoded))

# Get the byte values for the encoded string (integers from 0 to 255).
# One byte does not necessarily correspond to one Unicode character!
list_utf8_encoded = list(utf8_encoded)
print(f"\nList of byte values (int: 0 to 255): {list_utf8_encoded}")
print(len(list_utf8_encoded))

decoded_string = utf8_encoded.decode("utf-8")
print(f"\nDecoded string: {decoded_string}")

"""
Output:
Initial string: hello! こんにちは!
13

Bytes -- utf-8 encoded: b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
23

List of byte values (int: 0 to 255): [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
23

Decoded string: hello! こんにちは!
"""