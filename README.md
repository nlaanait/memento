```Memento```

Scalable multi-agent learning.
#### Install
```python
git clone git@code.ornl.gov:disMultiABM/memento.git
cd memento
pip install -e .
```

#### Structure
1. __agents.py__: neural nets to represent policies, value functions, etc... as well as containers like actor-critic.  
2. __utils.py__: various needed functions like discounted return calculations, buffers, etc...  
3. __logging.py__: loggers for different policy gradient algos.  
4. __optimizers.py__: optimization and update functions.  
