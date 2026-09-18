"""Классы запасного классификатора и текстовые описания для подсказок без обучения (zero-shot).

Описания на английском: SigLIP2 обучалась в основном на английских подписях.
"""

TYPES = {
    "самосвал": ["a dump truck with a tipper body", "a tipper truck at a weighbridge"],
    "бортовой грузовик": ["a flatbed cargo truck with side boards", "an old truck with a wooden cargo body"],
    "грузовик с прицепом": ["a truck pulling a trailer, a road train", "a grain truck with a trailer"],
    "седельный тягач": ["a semi-trailer truck, tractor unit with a semi-trailer", "an articulated lorry"],
    "трактор": ["a large wheeled farm tractor", "an articulated four-wheel-drive tractor pulling trailers"],
    "прицеп": ["the rear of a trailer, only the trailer is visible", "a farm trailer without the truck"],
    "легковой": ["a passenger car", "a sedan car"],
    "внедорожник/пикап": ["a pickup truck", "an SUV off-road vehicle"],
    "автобус/фургон": ["a bus", "a minibus or van"],
    "спецтехника": ["a front loader or excavator", "a combine harvester"],
}

MAKES = {
    "КамАЗ": ["a KAMAZ truck, Russian cab-over truck", "KamAZ 5511 or 65115 dump truck"],
    "МАЗ": ["a MAZ truck from Belarus, cab-over", "MAZ 5551 dump truck"],
    "ЗИЛ": ["a ZIL-130 Soviet truck with a long hood", "ZIL truck with a wide front grille"],
    "ГАЗ": ["a GAZ-53 Soviet truck with a rounded hood", "GAZ 3307 truck"],
    "Урал": ["a Ural 4320 all-terrain truck", "Ural truck with a big hood"],
    "Кировец": ["a Kirovets K-700 articulated tractor", "Kirovets K-744 tractor with a high cab"],
    "МТЗ (Беларус)": ["a Belarus MTZ-82 tractor", "MTZ Belarus 1221 wheeled tractor"],
    "LOVOL": ["a LOVOL tractor from China", "Foton Lovol wheeled tractor"],
    "John Deere": ["a John Deere green tractor", "John Deere combine"],
    "Shacman": ["a Shacman truck from China", "Shaanxi Shacman X3000 truck"],
    "Howo": ["a Sinotruk HOWO truck", "HOWO dump truck from China"],
    "FAW": ["a FAW truck from China", "FAW Jiefang truck"],
    "MAN": ["a MAN truck", "MAN TGS truck"],
    "Volvo": ["a Volvo truck", "Volvo FH truck"],
    "Mercedes-Benz": ["a Mercedes-Benz truck", "Mercedes Actros truck"],
    "Scania": ["a Scania truck", "Scania R series truck"],
    "DAF": ["a DAF truck", "DAF XF truck"],
    "Lada/ВАЗ": ["a Lada car", "a Soviet VAZ Zhiguli car"],
    "Toyota": ["a Toyota car or pickup", "Toyota Land Cruiser"],
}

# Ответы, которые разметчик может поставить вместо марки/типа.
UNKNOWN = "не видно / не определить"
OTHER = "другое"

TEMPLATE = "a CCTV photo of {} at a rural grain weighbridge"
