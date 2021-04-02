# Linux - Basics

Linux is a UNIX like operating system.  
Many high-performance computers run on Linux.

```bash
# list files
ls

# present working directory
pwd

# change directory
cd ..       # go up one
cd /path    # go to /path
cd          # go home
cd ~/folder # go to folder in home directory

# make directory
mkdir test

# copy
cp file
cp -r folder

# remove
rm file
rm -r folder

# move (rename)
mv filename folder/.
mv filename new_filename

# copy from server
scp username@server:/path/file .
scp -r folder username@server:/path/.

# echo variable
echo $USER

# for loop
for file in *.ipynb; do echo $file; done
    
# if statement
if [[ condition ]]; then
    something
else
    something_else
fi

# clone folder from server to client
rsync -v -P -r user@server:/source user@client:/destination
        
# find files
find -name "*md" -type f -size -2k

# search for pattern
grep -rnw '/path' -e 'pattern'

# find and replace string
sed -i 's/old/new/g' file

# connect to remote server
ssh username@server

# pipe commands
ls *.nc | grep 2015
```

Linux commands can be executed within a Jupyter Notebook using `!` at the start.

For more information, see this excellent [guide](https://github.com/cemacrr/linux_intro/blob/master/document.pdf) from Richard Rigby in CEMAC

