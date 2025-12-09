import speech_recognition as sr

print("Recherche des micros...")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone {index} : {name}")