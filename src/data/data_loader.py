import blpapi
import pandas as pd

# Définition de certains noms de champs BLP
DATE = blpapi.Name("date")
ERROR_INFO = blpapi.Name("errorInfo")
EVENT_TIME = blpapi.Name("EVENT_TIME")
FIELD_DATA = blpapi.Name("fieldData")
FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
FIELD_ID = blpapi.Name("fieldId")
SECURITY = blpapi.Name("security")
SECURITY_DATA = blpapi.Name("securityData")

class BLP():
    def __init__(self):
        """
        Initialisation de l'objet BLP et démarrage de la session.
        """
        self.session = blpapi.Session()

        # Démarrage de la session
        if not self.session.start():
            print("Échec de démarrage de la session.")
            return

        # Ouverture du service de données de référence ou sortie si impossible
        if not self.session.openService("//blp/refdata"):
            print("Impossible d'ouvrir //blp/refdata")
            return

        self.session.openService('//BLP/refdata')
        self.refDataSvc = self.session.getService('//BLP/refdata')

        print('Session ouverte')


    def bds(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):
            """
            Effectue une requête BDS (Bulk Data Service) pour récupérer les données demandées.

            Args:
                strSecurity (str ou list): Les titres pour lesquels les données sont demandées.
                strFields (str ou list): Les champs demandés.
                strOverrideField (str): Le champ à remplacer.
                strOverrideValue (str): La valeur de remplacement pour le champ spécifié.

            Returns:
                dict: Un dictionnaire contenant les données demandées.
            """
            # Création de la demande
            request = self.refDataSvc.createRequest('ReferenceDataRequest')

            # Assurer que les champs et les titres sont sous forme de listes
            if type(strFields) == str:
                strFields = [strFields]

            if type(strSecurity) == str:
                strSecurity = [strSecurity]

            # Ajout des champs à la demande
            for strD in strFields:
                request.append('fields', strD)

            # Ajout des titres à la demande
            for strS in strSecurity:
                request.append('securities', strS)

            # Ajout de la substitution de champ si nécessaire
            if strOverrideField != '':
                o = request.getElement('overrides').appendElement()
                o.setElement('fieldId', strOverrideField)
                o.setElement('value', strOverrideValue)

            # Envoi de la demande
            requestID = self.session.sendRequest(request)

            # Réception de la demande
            dict_Security_Fields = {}
            list_msg = []

            # Création de dictionnaires globaux pour stocker les données
            for field in strFields:
                globals()["dict_" + field] = {}

            while True:
                event = self.session.nextEvent()

                # Ignorer tout ce qui n'est pas une réponse partielle ou finale
                if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                        event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                    continue

                # Extraction du message de réponse
                msg = blpapi.event.MessageIterator(event).__next__()

                list_msg.append(msg)

                # Sortir de la boucle si la réponse est finale
                if event.eventType() == blpapi.event.Event.RESPONSE:
                    break

            # Extraction des données
            for msg in list_msg:
                for ticker in msg.getElement(SECURITY_DATA):
                    for field in strFields:
                        for field_data in ticker.getElement(FIELD_DATA):
                            for sub_field_data in field_data:
                                bloom_ticker = sub_field_data.getElement(0).getValue()

                                globals()['dict_' + field][bloom_ticker] = {}
                                for i in range(1, sub_field_data.numElements()):
                                    field_name = str(sub_field_data.getElement(i).name())

                                    try:
                                        globals()["dict_" + field][bloom_ticker][field_name] = sub_field_data.getElement(
                                            i).getValueAsFloat()
                                    except:
                                        globals()["dict_" + field][bloom_ticker][field_name] = sub_field_data.getElement(
                                            i).getValueAsString()

            # Conversion des données en DataFrame et stockage dans un dictionnaire
            for field in strFields:
                dict_Security_Fields[field] = pd.DataFrame.from_dict(globals()["dict_" + field], orient='index')

            return dict_Security_Fields

    def closeSession(self):
        """
        Fermeture de la session.
        """
        print("Session fermée")
        self.session.stop()

    def bdh(self, strSecurity, strFields, startdate, enddate, per='DAILY', perAdj='CALENDAR', days='NON_TRADING_WEEKDAYS', fill='PREVIOUS_VALUE', curr=None):
            """
            Récupère les données historiques pour un ensemble de titres et de champs.

            Args:
                strSecurity (list de str): Liste des tickers.
                strFields (list de str): Liste des champs, doivent être des champs statiques (par exemple, px_last au lieu de last_price).
                startdate (date): Date de début.
                enddate (date): Date de fin.
                per (str): Sélection de périodicité ; quotidienne, mensuelle, trimestrielle, semestrielle ou annuelle.
                perAdj (str): Ajustement de périodicité : ACTUAL, CALENDAR, FISCAL.
                curr (str): Devise, sinon la devise par défaut est utilisée.
                days (str): Option de remplissage des jours non négociables : NON_TRADING_WEEKDAYS*, ALL_CALENDAR_DAYS ou ACTIVE_DAYS_ONLY.
                fill (str): Méthode de remplissage des jours non négociables : PREVIOUS_VALUE, NIL_VALUE.

            Returns:
                dict: Un dictionnaire contenant les données demandées.
            """

            # Création de la demande
            request = self.refDataSvc.createRequest('HistoricalDataRequest')

            # Assurer que les champs et les titres sont sous forme de listes
            if type(strFields) == str:
                strFields = [strFields]

            if type(strSecurity) == str:
                strSecurity = [strSecurity]

            # Ajout des champs à la demande
            for strF in strFields:
                request.append('fields', strF)

            # Ajout des titres à la demande
            for strS in strSecurity:
                request.append('securities', strS)

            # Définition des autres paramètres
            request.set('startDate', startdate.strftime('%Y%m%d'))
            request.set('endDate', enddate.strftime('%Y%m%d'))
            request.set('periodicitySelection', per)
            request.set('periodicityAdjustment', perAdj)
            request.set('nonTradingDayFillOption', days)
            request.set('nonTradingDayFillMethod', fill)
            request.set('currency', curr)

            # Envoi de la demande
            requestID = self.session.sendRequest(request)

            # Réception de la demande
            dict_Security_Fields = {}

            list_msg = []

            for field in strFields:
                globals()["dict_" + field] = {}

            while True:
                event = self.session.nextEvent()

                # Ignorer tout ce qui n'est pas une réponse partielle ou finale
                if (event.eventType() != blpapi.event.Event.RESPONSE) & (event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                    continue  # Revenir au début de la boucle

                # Extraction des messages de réponse
                for msg in blpapi.event.MessageIterator(event):
                    list_msg.append(msg)

                # Sortir de la boucle si la réponse est finale
                if event.eventType() == blpapi.event.Event.RESPONSE:
                    break

            # Exploitation des données
            for msg in list_msg:
                ticker = str(msg.getElement(SECURITY_DATA).getElement(SECURITY).getValue())
                for field in strFields:
                    globals()['dict_' + field][ticker] = {}

                for field_data in msg.getElement(SECURITY_DATA).getElement(FIELD_DATA):
                    dat = field_data.getElement(0).getValue()

                    for i in range(1, field_data.numElements()):
                        field_name = str(field_data.getElement(i).name())
                        try:
                            globals()["dict_" + field_name][ticker][dat] = field_data.getElement(i).getValueAsFloat()
                        except:
                            globals()["dict_" + field_name][ticker][dat] = field_data.getElement(i).getValueAsString()

            for field in strFields:
                dict_Security_Fields[field] = pd.DataFrame.from_dict(globals()["dict_" + field], orient='index').T

            return dict_Security_Fields