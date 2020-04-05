proj_home=$(pwd)

git clone https://github.com/CantelopePeel/z3.git ./qq_z3
git clone https://github.com/CantelopePeel/QQ.git ./qq_proj
git clone https://github.com/CantelopePeel/QuEST.git ./qq_sim

cd qq_z3
git checkout master
cd ..

# Initialize and Activate Venv
python3 -m venv ./venv
source venv/bin/activate

cd qq_z3
python scripts/mk_make.py --python --prefix=$VIRTUAL_ENV -t
cd build
make
make install
cd ../..
python -c 'import z3; print(z3.get_version_string())'

cd ..


pip install -r requirements.txt

deactivate