from dataclasses import dataclass

from src.backend.modules.srs.abstract_srs import AbstractCard, AbstractDeck, AbstractSRS, CardState, Flag


@dataclass
class SrsAction:
    description: str
    result_object: AbstractCard | AbstractDeck | None = None

    @staticmethod
    def add_deck(srs: AbstractSRS, deck_name: str) -> "SrsAction":
        result_deck = srs.add_deck(deck_name)
        return SrsAction(description=f"Created deck: {deck_name}", result_object=result_deck)

    @staticmethod
    def rename_deck(srs: AbstractSRS, deck: AbstractDeck, new_name: str) -> "SrsAction":
        old_name = deck.name
        result_deck = srs.rename_deck(deck, new_name)

        return SrsAction(description=f"Renamed deck from {old_name} to {new_name}", result_object=result_deck)

    @staticmethod
    def delete_deck(srs: AbstractSRS, deck: AbstractDeck) -> "SrsAction":
        srs.delete_deck(deck)
        return SrsAction(description=f"Deleted deck: {deck.name}")

    @staticmethod
    def add_card(
        srs: AbstractSRS, deck: AbstractDeck, question: str, answer: str, flag: Flag, state: CardState
    ) -> "SrsAction":
        card = srs.add_card(deck, question, answer, flag, state)
        return SrsAction(
            description=f"Added card to deck {deck.name}: {card.question} - {card.answer}", result_object=card
        )

    @staticmethod
    def edit_card_question(srs: AbstractSRS, card: AbstractCard, new_question: str) -> "SrsAction":
        old_question = card.question
        new_card = srs.edit_card_question(card, new_question)
        return SrsAction(
            description=f"Edited card question from {old_question} to {new_question}", result_object=new_card
        )

    @staticmethod
    def edit_card_answer(srs: AbstractSRS, card: AbstractCard, new_answer: str) -> "SrsAction":
        old_answer = card.answer
        new_card = srs.edit_card_answer(card, new_answer)
        return SrsAction(
            description=f"Edited card answer of card {card.question} from {old_answer} to {new_answer}",
            result_object=new_card,
        )

    @staticmethod
    def edit_card_flag(srs: AbstractSRS, card: AbstractCard, new_flag: Flag) -> "SrsAction":
        old_flag = card.flag
        new_card = srs.edit_card_flag(card, new_flag)
        return SrsAction(
            description=f"Edited card flag of card {card.question} from {old_flag} to {new_flag}",
            result_object=new_card,
        )

    @staticmethod
    def edit_card_state(srs: AbstractSRS, card: AbstractCard, new_state: CardState) -> "SrsAction":
        old_state = card.state
        new_card = srs.edit_card_state(card, new_state)
        return SrsAction(
            description=f"Edited card state of card {card.question} from {old_state} to {new_state}",
            result_object=new_card,
        )

    @staticmethod
    def change_deck_of_card(srs: AbstractSRS, card: AbstractCard, new_deck: AbstractDeck) -> "SrsAction":
        old_deck = srs.get_deck_of_card(card)
        card = srs.change_deck_of_card(card, new_deck)
        return SrsAction(
            description=f"Changed deck of card {card.question} from {old_deck.name} to {new_deck.name}",
            result_object=card,
        )

    @staticmethod
    def copy_card_to(srs: AbstractSRS, card: AbstractCard, deck: AbstractDeck) -> "SrsAction":
        new_card = srs.copy_card_to(card, deck)
        return SrsAction(description=f"Copied card {card.question} to deck {deck.name}", result_object=new_card)

    @staticmethod
    def delete_card(srs: AbstractSRS, card: AbstractCard) -> "SrsAction":
        srs.delete_card(card)
        return SrsAction(description=f"Deleted card {card.question}", result_object=card)


class HistoryManager:
    def __init__(self):
        self.srs_action_history: list[SrsAction] = []
        self.latest_queries = []
        from src.backend.modules.ai_assistant.state_manager import ExecutionResult

        self.latest_execution_result: list[ExecutionResult] = []

    def add_action(self, action: SrsAction):
        self.srs_action_history.append(action)

    def get_string_history(self) -> str:
        combined_actions = []
        for action in self.srs_action_history:
            combined_actions.append({"description": action.description, "result_object": str(action.result_object)})
        return str(combined_actions)
