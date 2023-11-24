

def main():
    ans = 0
    n = 1000
    tmp = 1
    reward = 5
    discount = .99
    for i in range(n):
        ans += tmp * reward
        tmp *= .99
        if i % 100 == 0:
            print(f'{i}: {ans}')


if __name__ == "__main__":
    main()
