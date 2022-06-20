# ly
Generates a lyric video using DALL-E. This app is a tiny Python script that uses existing DALL-E
infrastructure to automate creation of images from song lyrics.

Steps to take:

    1) Clone https://github.com/saharmor/dalle-playground on a Linux machine. I used WSL on Windows.
    2) Follow the instructions for that repo to get your machine in a state where app.py works
       (though it's not used here)
    3) Go to the backend folder and copy ly.pt to that location.
    4) Get lyrics for the song you want. The repo has lucy.txt as an example. 
       Put the file in the backend folder.
       Perhaps use the ID app https://github.com/davidly/id to extract lyrics from flac/ MP3 files
    5) python ly.py lucy.txt
    6) This will take an hour or more on a CPU depending on the number of lyrics. 
       I spent two hours attempting to get a GPU working with JAX on WSL and failed. 
       Others have discovered this combination doesn't seem to work.
    7) PNG files will appear in the input filename _images folder, in this case lucy_images.
    8) A text file appears with the PNG files with _files appended to the song name.
       (here called lucy_files.txt).
    9) Use https://github.com/davidly/cv to create an MP4 slideshow: 
       cv /i:lucy_files.txt /w:256 /h:256 /o:lucy.mp4 /d:2000 /c
       sample output file lucy.mp4 is in this repo
    10) Or, use https://github.com/davidly/ic to create a collage with lyrics on the images
       ic lucy_files.txt /o:lucy.png /c /n
    
     
