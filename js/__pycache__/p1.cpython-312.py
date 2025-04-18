�
    �f�gu  �                   �  � d dl mZmZmZ d dlZd dlZd dlmZ  ee�      Z	d� Z
d� Zej                  j                   ej                  �       d�      Z e
e�      Ze	j#                  ddd	g�
�      d� �       Zedk(  re	j'                  d��       yy)�    )�Flask�render_template�requestN)�get_close_matchesc           
      �>  � i }	 t        | dd��      5 }t        j                  |�      }t        |�       |D ]4  }|d   j	                  �       }t        t        g d�|dd  �      �      }|||<   �6 	 d d d �       |S # 1 sw Y   |S xY w# t        $ r t        d�       Y |S w xY w)N�rzutf-8)�encodingr   )z	Soil TypezClimate RequirementszWatering FrequencyzFertilizer TypezPest Control MethodszHarvest TimezYield per AcrezMarket Price�   zError: CSV file not found.)	�open�csv�reader�next�lower�dict�zip�FileNotFoundError�print)�	file_path�data�filer   �row�pattern�	responsess          �2C:\Users\dines\OneDrive\Documents\Desktop\js\p1.py�load_chatbot_datar   	   s�   � ��D�,��)�S�7�3�t��Z�Z��%�F���L����a�&�,�,�.�� ��7� ���G�	"� �	� !*��W�� � 4� �K� 4� �K�� � ,��*�+��K�,�s/   �B �AA7�-B �7B�<B �B �B�Bc                 ��   � | j                  �       } t        | |j                  �       dd��      }|rE|d   }||   }d|j                  �       � d�dj	                  d� |j                  �       D �       �      z   S y	)
Nr
   g333333�?)�n�cutoffr   zInformation about z:
�
c              3   �0   K  � | ]  \  }}|� d |� ��� � y�w)z: N� )�.0�key�values      r   �	<genexpr>zget_response.<locals>.<genexpr>#   s%   � �� � E
�0A�*�#�u�s�e�2�e�W��0A�s   �zKI'm not sure I understand. Please ask about a specific crop or its details.)r   r   �keys�title�join�items)�
user_inputr   �matchr   r   s        r   �get_responser,      s   � ��!�!�#�J��j�$�)�)�+��3�G�E����(����M�	�#�G�M�M�O�#4�C�8�4�9�9� E
�09���0A�E
� <
� 
� 	
� Y�    zfarming_guidance_large.csv�/�GET�POST)�methodsc                  ��   � d} t         j                  dk(  r#t         j                  d   }t        |t        �      } t        dt         j                  j                  dd�      | ��      S )N� r0   r*   z
index.html)r*   �response)r   �method�formr,   �chatbot_datar   �get)r4   r*   s     r   �indexr9   -   sO   � ��H��~�~����\�\�,�/�
��
�L�9���<�G�L�L�4D�4D�\�SU�4V�ai�j�jr-   �__main__T)�debug)�flaskr   r   r   r   �os�difflibr   �__name__�appr   r,   �pathr(   �getcwdr   r7   �router9   �runr!   r-   r   �<module>rE      s�   �� 1� 1� 
� 	� %��H�o���(	Y� �G�G�L�L������&B�C�	� ��+�� ���3�����(�k� )�k� �z���G�G�$�G�� r-   