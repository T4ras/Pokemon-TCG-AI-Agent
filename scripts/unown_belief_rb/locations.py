"""Native storage and destination aliases, from Types.h/State.h/CardMove.h."""
from .common import UnsupportedEvent
AREA={1:'D',2:'H',3:'B',4:'A',5:'B',6:'P',7:'B',8:'B',9:'B',10:'B',11:'B',12:'L',13:'R',14:'D'}
NAMES={0:'All',1:'Deck',2:'Hand',3:'Trash',4:'Active',5:'Bench',6:'Prize',7:'Stadium',8:'Energy',9:'Tool',10:'PreEvolution',11:'Player',12:'Looking',13:'Playing',14:'DeckBottom',15:'Me',16:'Effected',17:'EffectedPreTarget',18:'SelectedList',19:'TriggerSubject',20:'TriggerObject',21:'Attach',22:'TurnPlay',23:'AttackPreMyTurn',24:'Temporary'}
def zone(value,source=False):
 if value not in AREA or source and value==14:raise UnsupportedEvent('Native area is not an implemented physical '+('source: ' if source else 'destination: ')+str(value))
 return AREA[value]
def destination_position(value):return 'bottom' if value==14 else 'top' if value==1 else 'append'
