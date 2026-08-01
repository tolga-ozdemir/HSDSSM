while read -r pkg; do
    # Strip everything after the second '=' or first '=' if only two parts exist
    # Example: yacs=0.1.8=h5eee -> yacs==0.1.8
    clean_name=$(echo "$pkg" | cut -d'=' -f1)
    clean_ver=$(echo "$pkg" | cut -d'=' -f2)
    
    if [ ! -z "$clean_ver" ]; then
        install_target="${clean_name}==${clean_ver}"
    else
        install_target="${clean_name}"
    fi

    echo "Installing: $install_target"
    pip install --no-cache-dir "$install_target" || echo "ERROR: Could not install $pkg"
done < pip_only_reqs.txt