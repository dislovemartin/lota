.PHONY: all build test deploy clean

all: build test deploy

build:
	bash scripts/build.sh

test:
	bash scripts/test_runner.sh

deploy:
	bash scripts/deploy.sh

clean:
	rm -rf build
	docker system prune -f

