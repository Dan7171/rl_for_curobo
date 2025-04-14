def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


if __name__ == "__main__":
    @static_vars(counter=0)
    def foo():
        foo.counter += 1
        print("Counter is %d" % foo.counter)

    foo()
    foo()
    foo()

    @static_vars(string='')
    def bar():
        bar.string += 'x'
        print(bar.string)

    bar()
    bar()