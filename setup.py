from setuptools import setup, find_packages

setup(
    name='Koramco',  # 설치될 패키지 이름
    version='0.1.0',  # 버전
    packages=find_packages(),  # 현재 디렉토리에서 패키지들을 찾아서 포함
    package_data={'Koramco':['AddTokens/*.csv']},
    install_requires=[  # 필요한 패키지 목록 (선택 사항)
        'requests',
    ],
    author='EXEM', # 작성자 이름
    author_email='', # 작성자 이메일
    description='', # 모듈 설명
    url='', # 모듈 저장소 URL (선택 사항)
    classifiers=[  # PyPI에 등록될 때 사용되는 메타데이터 (선택 사항)
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)