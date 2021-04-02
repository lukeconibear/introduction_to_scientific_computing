# Satellite data

## Download satellite data from NASA LAADS

1. Go to the [LAADS website](https://ladsweb.modaps.eosdis.nasa.gov/) and search for the data you're after.  
    - e.g. [MODIS Thermal Anomalies/Fire for 2020](https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/6/MOD14A1/2020/).  
    - e.g. [MODIS AOD for 2019](https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD04_L2/2019/).  
2. Create main download script as below and name it `laads-main-download.sh`:
    ```bash
    #!/bin/bash
    for julday in {001..365};
        do for satellite in 'MOD04_L2' 'MYD04_L2';
            do . laads-data-download.sh -s https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/$satellite/2019/$julday -d TARGET_PATH -t KEY;
        done;
    done
    ```
    - Change the satellite ID to the one you want (e.g. MOD14A1).
    - Change the collection number (e.g. 61 to 6)
    - Change the year.
    - Replace `TARGET_PATH` with where you want to save the data on your computer.
    - Replace `KEY` with your user key for LAADS.
        - You can get from [here](https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/).
3. Create data download script as below and name it `laads-data-download.sh`:
    ```bash
    #!/bin/bash

    function usage {
      echo "Usage:"
      echo "  $0 [options]"
      echo ""
      echo "Description:"
      echo "  This script will recursively download all files if they don't exist"
      echo "  from a LAADS URL and stores them to the specified path"
      echo ""
      echo "Options:"
      echo "    -s|--source [URL]         Recursively download files at [URL]"
      echo "    -d|--destination [path]   Store directory structure to [path]"
      echo "    -t|--token [token]        Use app token [token] to authenticate"
      echo ""
      echo "Dependencies:"
      echo "  Requires 'jq' which is available as a standalone executable from"
      echo "  https://stedolan.github.io/jq/download/"
    }

    function recurse {
      local src=$1
      local dest=$2
      local token=$3

      echo "Querying ${src}.json"

      for dir in $(curl -s -H "Authorization: Bearer ${token}" ${src}.json | jq '.[] | select(.size==0) | .name' | tr -d '"')
      do
        echo "Creating ${dest}/${dir}"
        mkdir -p "${dest}/${dir}"
        echo "Recursing ${src}/${dir}/ for ${dest}/${dir}"
        recurse "${src}/${dir}/" "${dest}/${dir}"
      done

      for file in $(curl -s -H "Authorization: Bearer ${token}" ${src}.json | jq '.[] | select(.size!=0) | .name' | tr -d '"')
      do
        if [ ! -f ${dest}/${file} ] 
        then
          echo "Downloading $file to ${dest}"
          # replace '-s' with '-#' below for download progress bars
          curl -s -H "Authorization: Bearer ${token}" ${src}/${file} -o ${dest}/${file}
        else
          echo "Skipping $file ..."
        fi
      done
    }

    POSITIONAL=()
    while [[ $# -gt 0 ]]
    do
      key="$1"

      case $key in
        -s|--source)
        src="$2"
        shift # past argument
        shift # past value
        ;;
        -d|--destination)
        dest="$2"
        shift # past argument
        shift # past value
        ;;
        -t|--token)
        token="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
      esac
    done

    if [ -z ${src+x} ]
    then 
      echo "Source is not specified"
      usage
      exit 1
    fi

    if [ -z ${dest+x} ]
    then 
      echo "Destination is not specified"
      usage
      exit 1
    fi

    if [ -z ${token+x} ]
    then 
      echo "Token is not specified"
      usage
      exit 1
    fi

    recurse "$src" "$dest" "$token"
    ```

4. Run the main download script: `. laads-main-download.sh`

## Merge