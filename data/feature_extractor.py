# ... (previous code: imports, constants, function definition)

    for i, genre_folder in enumerate(sorted(os.listdir(dataset_path))):
        genre_path = os.path.join(dataset_path, genre_folder)

        if os.path.isdir(genre_path):
            data["mapping"].append(genre_folder)
            print(f"\nProcessing genre: {genre_folder}")

            for filename in sorted(os.listdir(genre_path)):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)

                    try:
                        # Load the audio file
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        
                        # --- THIS IS THE NEW CODE BLOCK TO ADD ---

                        # We'll add a simple check to ensure the loaded signal is long enough
                        # to be split into our desired number of segments.
                        # SAMPLES_PER_TRACK was calculated as SAMPLE_RATE * TRACK_DURATION_SECONDS
                        if len(signal) >= SAMPLES_PER_TRACK:
                            
                            # Calculate the number of samples per segment
                            num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

                            # Iterate through the segments
                            for s in range(NUM_SEGMENTS):
                                # Calculate the start and end sample indices for the current segment
                                start_sample = s * num_samples_per_segment
                                end_sample = start_sample + num_samples_per_segment

                                # Extract the segment using NumPy array slicing. The 'segment' variable
                                # now holds a 3-second chunk of audio data.
                                segment = signal[start_sample:end_sample]

                                # The next tasks will involve extracting features from this 'segment'.
                                # For example:
                                # mfccs = librosa.feature.mfcc(y=segment, sr=sr)
                        # --- END OF THE NEW CODE BLOCK ---

                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue

    # ... (rest of the code)