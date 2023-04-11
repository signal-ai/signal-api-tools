import requests
import json
import backoff
from urllib.parse import urlparse, parse_qs


def response_to_url(response) -> str:
    """get the url from a response"""
    obj = urlparse(response.request.url)
    return f"{obj.scheme}://{obj.netloc}{obj.path}"


def response_to_params(response) -> dict:
    """get the params from a response"""
    obj = urlparse(response.request.url)
    return parse_qs(obj.query)


def response_to_body(response) -> dict:
    body = response.request.body
    return json.loads(body.decode()) if body is not None else {}


def fatal_code(e):
    # too many requets - slow down
    if e.response.status_code == 429:
        return False
    # not authorised - re authenticate
    if e.response.status_code == 401:
        return False
    # Fatal
    if 400 <= e.response.status_code < 600:
        return True
    return False


class Paginate:
    """
    A class to iterate over API requestes
    """

    def __init__(self, response):
        self.response = response

    def __iter__(self):
        return self

    def _get(self, response):
        """get the next get request from the previous one"""
        nxt = response.json().get("next-cursor", None)
        # the absence of next-cursor signifies we have reached the final page
        if not nxt:
            return None
        params = response_to_params(response)
        params["from-cursor"] = nxt
        return requests.request(
            "GET",
            response_to_url(response),
            headers=response.request.headers,
            params=params,
        )

    def _post(self, response):
        nxt = response.json().get("next-cursor", None)
        if not nxt:
            return None
        body = response_to_body(response)
        body["from-cursor"] = nxt
        return requests.request(
            "POST",
            response_to_url(response),
            headers=response.request.headers,
            json=body,
        )

    # retry requests with an exponentially increasing wait time upto 10 times
    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_value=10,
        giveup=fatal_code,
    )
    def __next__(self):
        """get the next response from the previous one"""
        response = self.response

        # Check if we have reached the final page
        if not response:
            raise StopIteration()

        # Check the latest response was valid
        response.raise_for_status()

        method = response.request.method
        if method == "GET":
            self.response = self._get(response)
        elif method == "POST":
            self.response = self._post(response)
        else:
            raise ValueError(f"{method} method not supported")

        return response


class Connection:
    def __init__(self, client_id, client_secret, url="https://api.signal-ai.com"):
        self._client_id = client_id
        self._client_secret = client_secret
        self._url = url
        self._temp_access_token = self._authenticate()

    def _authenticate(self):
        response = requests.post(
            f"{self._url}/auth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        return response.json().get("access_token")

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_value=10,
        giveup=fatal_code,
    )
    def _request(self, method, endpoint, params=None, json=None, additional_headers={}):
        """Make get requests using a tempory access token"""
        default_headers = {
            "Authorization": f"Bearer {self._temp_access_token}",
            "Content-Type": "application/json",
        }
        response = requests.request(
            method,
            f"{self._url}/{endpoint}",
            params=params,
            json=json,
            headers={**default_headers, **additional_headers},
        )

        # If the request is unauthorised try re-authenticating
        if response.status_code == 401:
            self._temp_access_token = self._authenticate()

        # Check the latest response was valid, if not raise an exception
        # and retry using backoff
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # attach API error messages to the exception
            raise requests.exceptions.HTTPError(
                "\n".join(
                    [str(e)]
                    + [errors[0] + " " + errors[1] for errors in e.response.json().get("errors", [])]
                ),
                response=e.response,
                request=e.request,
            )
        return response

    def entities(self, params):
        """Find the signal ID for entities using any combination of name and type"""
        response = self._request("GET", "entities", params)
        for page in Paginate(response):
            entities = page.json().get("entities")
            # we need an extra check here because of an API pagination bug
            if not entities:
                break
            for item in entities:
                yield item

    def topics(self, params):
        """Find the signal ID for A.I. trained topics by name"""
        response = self._request("GET", "topics", params)
        for page in Paginate(response):
            for item in page.json().get("topics"):
                yield item

    def sources(self, params):
        """Find the signal ID for publication sources"""
        response = self._request("GET", "sources", params)
        for page in Paginate(response):
            for item in page.json().get("sources"):
                yield item

    def get_source(self, source_id):
        return self._request("GET", f"sources/{source_id}").json()

    def source_locations(self):
        """Returns the list the regions, subregios and countries of publication"""
        response = self._request("GET", "source-locations", {})
        return response.json()


    def documents(self, document_id):
        """get a document by ID"""
        return self._request("GET", f"documents/{document_id}").json().get("document")

    def search(self, params):
        """Search for metadata about individual documents"""
        response = self._request("POST", "search", json=params)
        total = response.json().get("stats").get("total")
        # yield a lazy sequence. This could take a long time
        # and the user might not want all the results
        for page in Paginate(response):
            for item in page.json().get("documents", []):
                yield {
                    "document": item,
                    # include the length of the sequence
                    "stats": {"total": total},
                }

    def metrics(self, params):
        """aggregated metrics which can be sliced and diced along multiple dimensions:
        date, publication source, publication country, topics, entities, sentiment, etc.."""
        return self._request("POST", "metrics", json=params).json().get("aggregations")

    def affinity(self, params):
        """Leverage the power of the Signal knowledge graph that we construct from our content and update over time."""
        response = self._request("POST", "affinity", json=params)
        for page in Paginate(response):
            source_concept = page.json()["source-concept"]
            for item in page.json().get("results"):
                item["source-concept"] = source_concept
                yield item

    def events(self, params, headers={}):
        response = self._request("POST", "events", json=params, additional_headers=headers)
        for page in Paginate(response):
            for event in page.json().get("events", []):
                yield event

    def get_event(self, event_hash):
        return self._request("GET", f"events/{event_hash}").json().get("event")