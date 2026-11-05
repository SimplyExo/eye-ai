SUMMARY = "MediaMTX / rtsp-simple-server is a ready-to-use and zero-dependency server and proxy"
HOMEPAGE = "https://github.com/bluenviron/mediamtx"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=77fd2623bd5398430be5ce60489c2e81"

SRC_URI = " \
    git://github.com/bluenviron/mediamtx.git;branch=main;protocol=https \
    file://mediamtx.yml \
    file://auto.crt \
    file://auto.key \
"

SRCREV = "9540cb6c1d9f1e9351fcc4ded3afc249ffff22bc"

inherit go

do_compile[network] = "1"

do_compile() {
    export GOCACHE="${WORKDIR}/go-cache"
    export GOMODCACHE="${WORKDIR}/go-modcache"
    
    export GOOS="linux"
    export GOARCH="arm"
    export GOARM="6"
    export CGO_ENABLED="0"

    cd ${S}
    
    go generate ./...
    go build -trimpath -v -o mediamtx .
}

do_install() {
    install -d ${D}${sysconfdir}/mediamtx

    install -m 0644 ${UNPACKDIR}/mediamtx.yml ${D}${sysconfdir}/mediamtx/mediamtx.yml
    install -m 0600 ${UNPACKDIR}/auto.crt ${D}${sysconfdir}/mediamtx/auto.crt
    install -m 0600 ${UNPACKDIR}/auto.key ${D}${sysconfdir}/mediamtx/auto.key

    install -d ${D}${bindir}
    install -m 0755 ${S}/mediamtx ${D}${bindir}/mediamtx
}

FILES:${PN} += " \
    ${bindir}/mediamtx \
    ${sysconfdir}/mediamtx \
"
